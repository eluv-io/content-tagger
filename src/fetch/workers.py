import threading
import os
import tempfile
from loguru import logger
import shutil
from copy import deepcopy
from contextlib import contextmanager
import threading
import time

from common_ml.video_processing import unfrag_video
from common_ml.utils.files import get_file_type, encode_path
from requests import HTTPError

from src.fetch.model import *
from src.common.content import Content, ContentConfig
from src.common.errors import BadRequestError, MissingResourceError
from src.fetch.model import DownloadResult

StreamKey = tuple[str, str]

class FetchRateLimiter:
    def __init__(self, max_concurrent: int):
        self._sem = threading.Semaphore(max_concurrent)
        self._locks: dict[StreamKey, threading.Lock] = {}
        self._locks_mu = threading.Lock()

    def _get_lock(self, key: StreamKey) -> threading.Lock:
        with self._locks_mu:
            return self._locks.setdefault(key, threading.Lock())

    @contextmanager
    def permit(self, key: StreamKey):
        # throttle total API calls
        self._sem.acquire()
        try:
            # avoid downloading the same stream from two places
            lock = self._get_lock(key)
            with lock:
                yield
        finally:
            self._sem.release()

class VodWorker(FetchSession):
    def __init__(
            self,
            q: Content,
            scope: VideoScope,
            rate_limiter: FetchRateLimiter,
            meta: VideoMetadata, 
            ignore_parts: list[str],
            output_dir: str,
            exit: threading.Event | None = None
    ):
        self.q = q
        self.scope = scope
        self.rl = rate_limiter
        self.meta = meta
        self.output_dir = output_dir
        self.ignore_parts = set(ignore_parts)
        self.exit = exit

    def metadata(self) -> VideoMetadata:
        return deepcopy(self.meta)

    def download(self) -> DownloadResult:
        with self.rl.permit((self.q.qhit, str(self.scope.stream))):
            return self._download()

    @property
    def path(self) -> str:
        return self.output_dir

    def _download(self) -> DownloadResult:
        """
        Downloads the parts from the stream and returns them. 
        
        If req.replace is True, doesn't return already tagged tags.

        Returns:
            DownloadResult containing successful_sources and failed_part_hashes
        """

        if self.meta.codec_type not in ["video", "audio"]:
            raise BadRequestError(
                f"Invalid codec type for live: {self.meta.codec_type}. Must be 'video' or 'audio'."
            )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        tmp_path = tempfile.mkdtemp()

        successful_sources = []
        failed_parts = []

        scope = self.scope
        start_time, end_time = scope.start_time, scope.end_time

        to_download = self.meta.parts

        logger.info(f"Downloading stream {self.scope.stream} with {len(to_download)} total parts.")

        if self.ignore_parts:
            logger.info(f"Filtering {len(self.ignore_parts)} already tagged parts")

        for idx, part_hash in enumerate(to_download):
            if self.exit and self.exit.is_set():
                break

            if part_hash in self.ignore_parts:
                continue

            pstart = idx * self.meta.part_duration
            pend = (idx + 1) * self.meta.part_duration
            idx_str = str(idx).zfill(4)

            # Check if part is within time range
            if not (
                start_time <= pstart < end_time
            ) and not (start_time <= pend < end_time):
                continue

            filename = f"{idx_str}_{part_hash}{'.mp4' if self.meta.codec_type == 'video' else '.m4a'}"
            save_path = os.path.join(self.output_dir, filename)

            if os.path.exists(save_path):
                source = Source(
                    name=part_hash,
                    filepath=save_path,
                    offset=int(pstart * 1000),
                    wall_clock=None
                )
                successful_sources.append(source)
                continue

            tmpfile = os.path.join(tmp_path, f"{idx_str}_{part_hash}")

            try:
                self.q.download_part(save_path=tmpfile, part_hash=part_hash)

                if self.meta.codec_type == "video":
                    unfrag_video(tmpfile, save_path)
                else:
                    shutil.move(tmpfile, save_path)

                source = Source(
                    name=part_hash,
                    filepath=save_path,
                    offset=int(pstart * 1000),
                    wall_clock=None
                )
                successful_sources.append(source)

            except Exception as e:
                if os.path.exists(save_path):
                    # Remove the corrupt file if it exists
                    os.remove(save_path)
                failed_parts.append(part_hash)
                logger.error(
                    f"Failed to download part {part_hash} for {self.q.qhit}: {str(e)}"
                )
                continue

            # check that length of the file is equal to the part length
            # if not last_part and self.meta.codec_type == "video":
            #     actual_duration = get_video_length(save_path)
            #     assert abs(actual_duration - self.meta.part_duration) < 1e-3

            # TODO: check for audio as well. 

        shutil.rmtree(tmp_path, ignore_errors=True)

        return DownloadResult(
            sources=successful_sources, 
            failed=failed_parts,
            done=True
        )

class AssetWorker(FetchSession):
    def __init__(
            self,
            q: Content,
            scope: AssetScope,
            rate_limiter: FetchRateLimiter,
            meta: AssetMetadata,
            ignore_assets: list[str],
            output_dir: str,
            exit: threading.Event | None = None
    ):
        self.q = q
        self.scope = scope
        self.rl = rate_limiter
        self.meta = meta
        self.output_dir = output_dir
        self.ignore_assets = set(ignore_assets)
        self.exit = exit

    def metadata(self) -> AssetMetadata:
        return deepcopy(self.meta)
    
    def download(self) -> DownloadResult:
        with self.rl.permit((self.q.qhit, "assets")):
            return self._download()

    @property
    def path(self) -> str:
        return self.output_dir

    def _download(self) -> DownloadResult:
        """
        Downloads asset files (images) from the content object.
        
        Returns:
            DownloadResult containing successful_sources and failed asset names
        """
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        scope = self.scope

        # Get list of assets to download
        if scope.assets is None:
            assets_meta = self.q.content_object_metadata(metadata_subtree='assets')
            assert isinstance(assets_meta, dict)
            assets_meta = list(assets_meta.values())
            assets = []
            for ameta in assets_meta:
                filepath = ameta.get("file")["/"]
                assert filepath.startswith("./files/")
                # strip leading term
                filepath = filepath[8:]
                assets.append(filepath)
        else:
            assets = scope.assets

        total_assets = len(assets)
        assets = [asset for asset in assets if get_file_type(asset) == "image"]
        logger.info(f"Found {len(assets)} image assets out of {total_assets} assets for {self.q.qhit}")
        
        if len(assets) == 0:
            return DownloadResult(
                sources=[],
                failed=[],
                done=True
            )

        if self.ignore_assets:
            logger.info(f"Filtering {len(self.ignore_assets)} already tagged assets")

        # Filter out already tagged assets
        assets_to_process = [asset for asset in assets if asset not in self.ignore_assets]
        
        # Separate assets that already exist vs need downloading
        already_downloaded = []
        to_download = []
        
        for asset in assets_to_process:
            if self.exit and self.exit.is_set():
                break
                
            save_path = os.path.join(self.output_dir, encode_path(asset))
            if os.path.exists(save_path):
                already_downloaded.append(asset)
            else:
                to_download.append((asset, save_path))

        if already_downloaded:
            logger.info(f"{len(already_downloaded)} assets already retrieved for {self.q.qhit}")

        logger.info(f"{len(to_download)} assets need to be downloaded for {self.q.qhit}")

        # Download new assets
        successful_sources = []
        failed_assets = []

        if to_download and not (self.exit and self.exit.is_set()):
            newly_downloaded = self._download_concurrent(to_download)
            
            # Create sources for newly downloaded assets
            for asset in newly_downloaded:
                source = Source(
                    name=asset,
                    filepath=os.path.join(self.output_dir, encode_path(asset)),
                    offset=0,
                    wall_clock=None
                )
                successful_sources.append(source)
            
            # Track failed downloads
            assets_to_download = [asset for asset, _ in to_download]
            failed_assets = list(set(assets_to_download) - set(newly_downloaded))

        # Create sources for already downloaded assets
        for asset in already_downloaded:
            source = Source(
                name=asset,
                filepath=os.path.join(self.output_dir, encode_path(asset)),
                offset=0,
                wall_clock=None
            )
            successful_sources.append(source)

        return DownloadResult(
            sources=successful_sources,
            failed=failed_assets,
            done=True
        )

    def _download_concurrent(self, file_jobs: list[tuple[str, str]]) -> list[str]:
        """Download multiple files concurrently using the content API"""
        status = self.q.download_files(dest_path=self.output_dir, file_jobs=file_jobs)
        assert isinstance(status, list) and len(status) == len(file_jobs)
        return [asset for (asset, _), error in zip(file_jobs, status) if error is None]

class LiveWorker(FetchSession):
    def __init__(
            self,
            q: Content,
            scope: LiveScope,
            rate_limiter: FetchRateLimiter,
            meta: LiveMetadata,
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
    
    def metadata(self) -> LiveMetadata:
        return deepcopy(self.meta)
    
    @property
    def path(self) -> str:
        return self.output_dir
    
    def download(self) -> DownloadResult:
        with self.rl.permit((self.q.qhit, str(self.scope.stream))):
            return self._download()
    
    def _download(self) -> DownloadResult:
        """
        Downloads a single segment from a live stream.
        
        Returns:
            DownloadResult containing the downloaded segment.
            done is always False for live streams (they never end).
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
        
        # TODO: filename weirdness on first segment
        filename = f"segment_{chunk_size}_{str(idx).zfill(4)}.mp4"
        save_path = os.path.join(self.output_dir, filename)
        
        segment_info = self.q.live_media_segment(
            object_id=self.q.qhit,
            dest_path=save_path,
            segment_idx=idx,
            segment_length=chunk_size,
        )

        seg_offset = segment_info.seg_offset_millis
        seg_idx = segment_info.seg_num
        seg_size = segment_info.actual_duration * 1000
        wall_clock = segment_info.seg_time_epoch_millis
        
        source = Source(
            name=f"segment_{chunk_size}_{idx}",
            filepath=save_path,
            offset=seg_offset,
            wall_clock=wall_clock
        )

        logger.info(
            f"Downloaded live segment {seg_idx} for {self.q.qhit}",
            extra={"segment": seg_idx, "offset_sec": seg_offset / 1000, "seg_size_sec": seg_size / 1000}
        )

        self.next_idx = seg_idx + 1

        if self.scope.max_duration is not None \
            and seg_offset + seg_size >= self.scope.max_duration * 1000:
            logger.info(f"Reached max duration of {self.scope.max_duration} seconds for live stream {self.q.qhit}")
            return DownloadResult(
                sources=[source],
                failed=[],
                done=True
            )

        return DownloadResult(
            sources=[source],
            failed=[],
            done=False  # Live streams never end
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
        # derive new Content with edge_token qhit
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
        with self.rl.permit((self.q.qhit, str(self.scope.stream))):
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
            logger.error(f"Failed to fetch live metadata for {self.q.qhit}: {str(e)}")
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
                logger.warning(f"No parts available yet for live stream {self.q.qhit}")
                return DownloadResult(
                    sources=[],
                    failed=[],
                    done=False
                )
            
            self.current_idx = len(current_parts) - 1
            self.last_known_parts = current_parts
            part_hash = current_parts[self.current_idx]
            
            logger.info(f"First call: downloading part {self.current_idx} (last available) for {self.q.qhit}")
            return self._download_part(part_hash, self.current_idx)
        
        # Subsequent calls: check if there's a new part
        if len(current_parts) > len(self.last_known_parts):
            logger.info(f"Detected new parts for live stream {self.q.qhit}")
            # New parts available, move to next index
            self.current_idx += 1
            self.last_known_parts = current_parts
            
            if self.current_idx < len(current_parts):
                part_hash = current_parts[self.current_idx]
                logger.info(f"New part available: downloading part {self.current_idx} for {self.q.qhit}")
                return self._download_part(part_hash, self.current_idx)
        else:
            logger.info(f"No new parts available yet for {self.q.qhit}")
        
        # No new parts available yet
        logger.debug(f"No new parts available for {self.q.qhit}, returning empty result")
        
        # Check if we've reached max duration
        if self.scope.max_duration is not None:
            elapsed_time = self.current_idx * self.meta.part_duration
            if elapsed_time >= self.scope.max_duration:
                logger.info(f"Reached max duration of {self.scope.max_duration}s for live stream {self.q.qhit}")
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
            logger.error(f"Failed to download part {part_hash} for {self.qwt.qhit}: {str(e)}")

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
                f"Failed to retrieve periods for live recording {self.qwt.qhit}"
            ) from e

        assert isinstance(periods, list)

        if len(periods) == 0:
            raise MissingResourceError(f"Live recording {self.qwt.qhit} is empty")

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
                f"Failed to retrieve live stream metadata from {self.qwt.qhit}"
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
                    f"Failed to retrieve live stream metadata from {self.qwt.qhit}"
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