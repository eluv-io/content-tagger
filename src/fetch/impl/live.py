
from copy import deepcopy
import os
import shutil
import tempfile
import threading
from requests.exceptions import HTTPError

from common_ml.video_processing import unfrag_video

from src.common.content import Content, ContentConfig
from src.common.errors import BadRequestError, MissingResourceError
from src.fetch.model import DownloadResult, FetchSession, LiveMetadata, LiveScope, Source, VideoMetadata
from src.fetch.rate_limit import FetchRateLimiter
from src.fetch.video_process import center_segment
from src.common.logging import logger

logger = logger.bind(module="live fetching")

def _get_live_source_name(chunk_size: int, stream_name: str, idx: int) -> str:
    return f"{stream_name}:segment_{chunk_size}_{idx}"

class LiveWorker(FetchSession):
    def __init__(
        self,
        q: Content,
        scope: LiveScope,
        rate_limiter: FetchRateLimiter,
        meta: LiveMetadata,
        ignore_sources: list[str],
        output_dir: str,
        exit: threading.Event | None = None
    ):
        self.q = q
        self.scope = scope
        self.rl = rate_limiter
        self.meta = meta
        self.output_dir = output_dir
        self.exit = exit
        self.next_idx = 0
        self.ignore_sources = set(ignore_sources)
    
    def metadata(self) -> LiveMetadata:
        return deepcopy(self.meta)
    
    @property
    def path(self) -> str:
        return self.output_dir
    
    def download(self) -> DownloadResult:
        with self.rl.permit((self.q.qid, str(self.scope.stream))):
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

        segment_info = self.q.live_media_segment(
            object_id=self.q.qid,
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
            f"Downloaded live segment {seg_idx} for {self.q.qid}",
            segment_idx=seg_idx, offset_sec=seg_offset / 1000, seg_size_sec=seg_size / 1000, wall_clock=wall_clock / 1000
        )

        self.next_idx = seg_idx + 1

        if self.scope.max_duration is not None \
            and seg_offset >= self.scope.max_duration * 1000:
            logger.info(f"Reached max duration of {self.scope.max_duration} seconds for live stream {self.q.qid}")
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

class LivePartWorker(FetchSession):
    def __init__(
        self,
        q: Content,
        scope: LiveScope,
        rate_limiter: FetchRateLimiter,
        output_dir: str,
        exit: threading.Event | None = None
    ):
        self.q = q
        # derive new Content with edge_token qid
        edge_token = q.content_object_metadata(
            metadata_subtree="live_recording/status/edge_write_token",
            resolve_links=False,
        )
        assert isinstance(edge_token, str)
        self.qwt = Content(
            edge_token, 
            auth=q._client.token,
            # NOTE: hardcoded for now
            cfg=ContentConfig(
                parts_url="https://host-154-14-185-98.contentfabric.io/config?self&qspace=main",
                config_url="https://host-154-14-185-98.contentfabric.io/config?self&qspace=main",
                live_media_url="https://host-76-74-34-204.contentfabric.io/config?self&qspace=main"
            )
        )
        self.scope = scope
        self.rl = rate_limiter
        self.meta = self._fetch_livestream_metadata(scope.stream)
        self.output_dir = output_dir
        self.exit = exit
        self.current_idx = -1  # Start at -1 so first call returns last part
        self.last_known_parts = []
    
    def metadata(self) -> VideoMetadata:
        return deepcopy(self.meta)
    
    @property
    def path(self) -> str:
        return self.output_dir
    
    def download(self) -> DownloadResult:
        with self.rl.permit((self.q.qid, str(self.scope.stream))):
            return self._download()
    
    def _download(self) -> DownloadResult:
        """
        Downloads parts from a live recording as they become available.
        
        First call: Returns the last available part
        Subsequent calls: Returns the next new part if available, otherwise empty result
        
        Returns:
            DownloadResult with the next available part or empty if no new parts
        """
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if self.exit and self.exit.is_set():
            return DownloadResult(
                sources=[],
                failed=[],
                done=True
            )
        
        # Fetch latest metadata to see if new parts are available
        try:
            latest_meta = self._fetch_livestream_metadata(self.scope.stream)
        except Exception as e:
            logger.error(f"Failed to fetch live metadata for {self.q.qid}: {str(e)}")
            return DownloadResult(
                sources=[],
                failed=[],
                done=False
            )
        
        # Update metadata
        self.meta = latest_meta
        current_parts = latest_meta.parts
        
        # First call: return last part
        if self.current_idx == -1:
            if len(current_parts) == 0:
                logger.warning(f"No parts available yet for live stream {self.q.qid}")
                return DownloadResult(
                    sources=[],
                    failed=[],
                    done=False
                )
            
            self.current_idx = len(current_parts) - 1
            self.last_known_parts = current_parts
            part_hash = current_parts[self.current_idx]
            
            logger.info(f"First call: downloading part {self.current_idx} (last available) for {self.q.qid}")
            return self._download_part(part_hash, self.current_idx)
        
        # Subsequent calls: check if there's a new part
        if len(current_parts) > len(self.last_known_parts):
            logger.info(f"Detected new parts for live stream {self.q.qid}")
            # New parts available, move to next index
            self.current_idx += 1
            self.last_known_parts = current_parts
            
            if self.current_idx < len(current_parts):
                part_hash = current_parts[self.current_idx]
                logger.info(f"New part available: downloading part {self.current_idx} for {self.q.qid}")
                return self._download_part(part_hash, self.current_idx)
        else:
            logger.info(f"No new parts available yet for {self.q.qid}")
        
        # No new parts available yet
        logger.debug(f"No new parts available for {self.q.qid}, returning empty result")
        
        # Check if we've reached max duration
        if self.scope.max_duration is not None:
            elapsed_time = self.current_idx * self.meta.part_duration
            if elapsed_time >= self.scope.max_duration:
                logger.info(f"Reached max duration of {self.scope.max_duration}s for live stream {self.q.qid}")
                return DownloadResult(
                    sources=[],
                    failed=[],
                    done=True
                )
        
        return DownloadResult(
            sources=[],
            failed=[],
            done=False
        )
    
    def _download_part(self, part_hash: str, idx: int) -> DownloadResult:
        """Download a single part and return it as a Source"""
        idx_str = str(idx).zfill(4)
        filename = f"{idx_str}_{part_hash}{'.mp4' if self.meta.codec_type == 'video' else '.m4a'}"
        save_path = os.path.join(self.output_dir, filename)
        
        # Check if already downloaded
        if os.path.exists(save_path):
            logger.info(f"Part {part_hash} already downloaded at {save_path}")
            offset = idx * self.meta.part_duration
            source = Source(
                name=part_hash,
                filepath=save_path,
                offset=int(offset * 1000),
                wall_clock=None
            )
            return DownloadResult(
                sources=[source],
                failed=[],
                done=False
            )
        
        # Download the part
        tmp_path = tempfile.mkdtemp()
        tmpfile = os.path.join(tmp_path, f"{idx_str}_{part_hash}")
        
        try:
            self.qwt.download_part(save_path=tmpfile, part_hash=part_hash)
            
            if self.meta.codec_type == "video":
                unfrag_video(tmpfile, save_path)
            else:
                shutil.move(tmpfile, save_path)
            
            offset = idx * self.meta.part_duration
            source = Source(
                name=part_hash,
                filepath=save_path,
                offset=int(offset * 1000),
                wall_clock=None
            )
            
            logger.info(f"Successfully downloaded part {idx} ({part_hash}) at offset {offset}s")
            
            return DownloadResult(
                sources=[source],
                failed=[],
                done=False
            )
            
        except Exception as e:
            logger.error(f"Failed to download part {part_hash} for {self.qwt.qid}: {str(e)}")

            if os.path.exists(save_path):
                os.remove(save_path)
            
            return DownloadResult(
                sources=[],
                failed=[part_hash],
                done=False
            )
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
    
    def _fetch_livestream_metadata(
        self, stream_name: str
    ) -> VideoMetadata:
        """Fetches metadata for livestream content."""
        try:
            periods = self.qwt.content_object_metadata(
                metadata_subtree="live_recording/recordings/live_offering",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(
                f"Failed to retrieve periods for live recording {self.qwt.qid}"
            ) from e

        assert isinstance(periods, list)

        if len(periods) == 0:
            raise MissingResourceError(f"Live recording {self.qwt.qid} is empty")

        stream = (
            periods[-1]
            .get("sources", {})
            .get(stream_name, {})
            .get("parts", [])
        )
        if len(stream) == 0:
            raise MissingResourceError(
                f"Stream {stream_name} was found in live recording, but no parts were found."
            )

        if stream_name == "video":
            codec_type = "video"
        elif stream_name.startswith("audio"):
            codec_type = "audio"
        else:
            raise BadRequestError(
                f"Invalid stream name for live: {stream_name}. Must be 'video' or start with prefix 'audio'."
            )

        try:
            xc_params = self.qwt.content_object_metadata(
                metadata_subtree="live_recording/recording_config/recording_params/xc_params",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(
                f"Failed to retrieve live stream metadata from {self.qwt.qid}"
            ) from e

        fps = None
        if codec_type == "video":
            try:
                live_stream_info = self.qwt.content_object_metadata(
                    metadata_subtree="live_recording_config/probe_info/streams",
                    resolve_links=False,
                )
            except HTTPError as e:
                raise HTTPError(
                    f"Failed to retrieve live stream metadata from {self.qwt.qid}"
                ) from e
            
            assert isinstance(live_stream_info, list)

            video_stream_info = None
            for stream_info in live_stream_info:
                if stream_info["codec_type"] == "video":
                    video_stream_info = stream_info
                    break

            if video_stream_info is None:
                raise MissingResourceError(
                    "Video stream not found in live stream metadata"
                )

            fps = self._parse_fps(video_stream_info["frame_rate"])
            assert isinstance(xc_params, dict)
            part_duration = xc_params.get("seg_duration", None)

            if part_duration is None:
                raise MissingResourceError(
                    "Part duration not found in live stream metadata"
                )
            part_duration = float(part_duration)
        else:
            assert isinstance(xc_params, dict)
            sr = xc_params.get("sample_rate", None)
            ts = xc_params.get("audio_seg_duration_ts", None)

            if sr is None or ts is None:
                raise MissingResourceError(
                    "Sample rate or audio segment duration not found in live stream metadata"
                )

            part_duration = int(ts) / int(sr)

        # Filter out parts with finalization_time == 0, meaning the part is still live.
        parts = [
            part["hash"]
            for part in stream
            if part["finalization_time"] != 0 and part["size"] > 0
        ]

        return VideoMetadata(
            parts=parts, part_duration=part_duration, fps=fps, codec_type=codec_type
        )

    def _parse_fps(self, frame_rate: str) -> float:
        """Parse FPS from string like '30/1' or '30000/1001'"""
        if '/' in frame_rate:
            num, denom = frame_rate.split('/')
            return float(num) / float(denom)
        return float(frame_rate)