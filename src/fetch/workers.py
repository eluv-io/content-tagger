import threading
import os
import tempfile
from loguru import logger
import shutil
from copy import deepcopy
from contextlib import contextmanager
import threading

from common_ml.video_processing import unfrag_video
from common_ml.utils.files import get_file_type, encode_path

from src.fetch.model import *
from src.common.content import Content
from src.common.errors import BadRequestError
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
                    offset=pstart,
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
                    offset=pstart,
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
                    offset=0
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
                offset=0
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
        self.call_count = 0
    
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
        idx = self.call_count
        
        # Create filename for this segment
        filename = f"segment_{chunk_size}_{str(idx).zfill(4)}.mp4"
        save_path = os.path.join(self.output_dir, filename)
        
        # Download the live segment
        # Note: We ignore segment_idx and segment_length for now as per instructions
        segment_info = self.q.live_media_segment(
            object_id=self.q.qhit,
            dest_path=save_path
        )
        
        # Override the segment info as per instructions
        # Use call_count for calculations
        offset = self.call_count * chunk_size
        
        source = Source(
            name=f"segment_{chunk_size}_{idx}",
            filepath=save_path,
            offset=float(offset)
        )
        
        self.call_count += 1
        
        logger.info(
            f"Downloaded live segment {idx} for {self.q.qhit}",
            extra={"segment": idx, "offset": offset, "chunk_size": chunk_size}
        )

        if self.scope.max_duration is not None \
            and offset >= self.scope.max_duration:
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