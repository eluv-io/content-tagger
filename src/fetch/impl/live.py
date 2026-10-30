
from copy import deepcopy
import os
import threading
import time

from common_ml.video_processing import get_fps

from src.common.content import QAPI
from src.fetch.model import DownloadResult, FetchSession, LiveScope, Source, MediaMetadata
from src.fetch.rate_limit import FetchRateLimiter
from src.fetch.video_process import center_segment
from src.common.logging import logger

logger = logger.bind(module="live fetching")

STREAM_POLL_INTERVAL = 5
# duration of a finalized live part (in seconds)
LIVE_PART_DURATION = 30.03

def _get_live_source_name(chunk_size: int, stream_name: str, idx: int) -> str:
    return f"{stream_name}:segment_{chunk_size}_{idx}"

class LiveWorker(FetchSession):
    def __init__(
        self,
        qapi: QAPI,
        scope: LiveScope,
        rate_limiter: FetchRateLimiter,
        meta: MediaMetadata,
        ignore_sources: list[str],
        output_dir: str,
        exit: threading.Event | None = None
    ):
        self.qapi = qapi
        self.scope = scope
        self.rl = rate_limiter
        self.meta = meta
        self.output_dir = output_dir
        self.exit = exit
        self.next_idx = 0
        self.ignore_sources = set(ignore_sources)
        self._stream_started = False
        self._created_at = time.time()
        # absolute pts of the recording period start
        self._start_pts: int | None = None
    
    def metadata(self) -> MediaMetadata:
        return deepcopy(self.meta)
    
    @property
    def path(self) -> str:
        return self.output_dir
    
    def download(self) -> DownloadResult:
        if not self._wait_for_stream():
            return DownloadResult(sources=[], failed=[], done=False)
        if self._start_pts is None:
            self._start_pts = self._fetch_start_pts()
        with self.rl.permit((self.qapi.id(), str(self.scope.stream))):
            return self._download()

    def _wait_for_stream(self) -> bool:
        """Returns True once the stream is running, otherwise sleeps for a poll interval and returns False.

        Lets a job (and its container) start before the livestream does.
        """
        if self._stream_started:
            return True

        if self.qapi.is_live_running():
            logger.info(f"Livestream {self.qapi.id()} is running")
            if self.scope.start_at_live_edge:
                self.next_idx = self._live_edge_idx()
            self._stream_started = True
            return True

        waited = time.time() - self._created_at
        if waited > self.scope.start_timeout:
            raise TimeoutError(f"Livestream {self.qapi.id()} did not start within {self.scope.start_timeout} seconds")

        logger.debug(f"Waiting for livestream {self.qapi.id()} to start", waited_sec=int(waited))
        if self.exit:
            self.exit.wait(STREAM_POLL_INTERVAL)
        else:
            time.sleep(STREAM_POLL_INTERVAL)
        return False
    
    def _fetch_start_pts(self) -> int | None:
        """Absolute pts of the current recording period start, or None if not yet known."""
        try:
            status = self.qapi.live_status()
            pts = status["custom"]["status"]["video"]["first_key_frame_pts"]
        except Exception as e:
            logger.opt(exception=e).warning(f"Failed to get period start pts for {self.qapi.id()}")
            return None
        if not isinstance(pts, int) or pts == 0:
            return None
        logger.info(f"Livestream {self.qapi.id()} period start pts", start_pts=pts)
        return pts

    def _live_edge_idx(self) -> int:
        """Index of the segment at the current live edge."""
        status = self.qapi.live_status()
        finalized_parts = status["custom"]["status"][self.scope.stream]["finalized_parts"]
        edge_sec = finalized_parts * LIVE_PART_DURATION
        idx = int(edge_sec // self.scope.segment_length)
        logger.info(f"Starting livestream {self.qapi.id()} at live edge", edge_sec=edge_sec, segment_idx=idx)
        return idx

    def _download(self) -> DownloadResult:
        """
        Downloads a single segment from a live stream.
        
        Returns:
            DownloadResult containing the downloaded segment.
        """
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if self.exit and self.exit.is_set():
            return DownloadResult(
                sources=[],
                failed=[],
                done=True
            )
        
        chunk_size = self.scope.segment_length

        idx = self.next_idx
        source_name = _get_live_source_name(chunk_size, self.scope.stream, idx)

        while source_name in self.ignore_sources:
            idx += 1
            source_name = _get_live_source_name(chunk_size, self.scope.stream, idx)
        
        filename = f"segment_{chunk_size}_{self.scope.stream}_{str(idx).zfill(4)}.mp4"
        save_path = os.path.join(self.output_dir, filename)

        segment_info = self.qapi.live_media_segment(
            object_id=self.qapi.id(),
            dest_path=save_path,
            segment_idx=idx,
            segment_length=chunk_size,
            stream=self.scope.stream
        )

        if self.scope.stream == "video":
            # ideally we can do this in the API just in case we have a different stream name for video
            center_segment(save_path)
            if self.meta.fps is None:
                self.meta.fps = get_fps(save_path)

        seg_offset = segment_info.seg_offset_millis
        seg_idx = segment_info.seg_num
        seg_size = segment_info.actual_duration * 1000
        wall_clock = segment_info.seg_time_epoch_millis
        pts = None
        if self._start_pts is not None and segment_info.seg_offset_ts is not None:
            pts = self._start_pts + segment_info.seg_offset_ts

        source = Source(
            name=source_name,
            filepath=save_path,
            offset=seg_offset,
            wall_clock=wall_clock,
            pts=pts
        )

        logger.info(
            f"Downloaded live segment {seg_idx} for {self.qapi.id()}",
            segment_idx=seg_idx, offset_sec=seg_offset / 1000, seg_size_sec=seg_size / 1000, wall_clock=wall_clock / 1000, pts=pts
        )

        self.next_idx = seg_idx + 1

        if self.scope.max_duration is not None \
            and seg_offset >= self.scope.max_duration * 1000:
            logger.info(f"Reached max duration of {self.scope.max_duration} seconds for live stream {self.qapi.id()}")
            return DownloadResult(
                sources=[],
                failed=[],
                done=True
            )

        # add to the total list of sources so we can track the total number of segments for status reporting
        self.meta.sources.append(source_name)

        return DownloadResult(
            sources=[source],
            failed=[],
            done=False
        )