
from copy import deepcopy
import os
import threading

from src.common.content import QAPI
from src.fetch.model import DownloadResult, FetchSession, LiveScope, Source, MediaMetadata
from src.fetch.rate_limit import FetchRateLimiter
from src.fetch.video_process import center_segment
from src.common.logging import logger

logger = logger.bind(module="live fetching")

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
    
    def metadata(self) -> MediaMetadata:
        return deepcopy(self.meta)
    
    @property
    def path(self) -> str:
        return self.output_dir
    
    def download(self) -> DownloadResult:
        with self.rl.permit((self.qapi.id(), str(self.scope.stream))):
            return self._download()
    
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
        
        chunk_size = self.scope.chunk_size

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

        seg_offset = segment_info.seg_offset_millis
        seg_idx = segment_info.seg_num
        seg_size = segment_info.actual_duration * 1000
        wall_clock = segment_info.seg_time_epoch_millis

        source = Source(
            name=source_name,
            filepath=save_path,
            offset=seg_offset,
            wall_clock=wall_clock
        )

        logger.info(
            f"Downloaded live segment {seg_idx} for {self.qapi.id()}",
            segment_idx=seg_idx, offset_sec=seg_offset / 1000, seg_size_sec=seg_size / 1000, wall_clock=wall_clock / 1000
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

        return DownloadResult(
            sources=[source],
            failed=[],
            done=False
        )