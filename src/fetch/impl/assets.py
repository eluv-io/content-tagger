

import os
import threading
from copy import deepcopy
from loguru import logger

from common_ml.utils.files import get_file_type, encode_path
from src.common.content import Content
from src.fetch.model import AssetMetadata, AssetScope, DownloadResult, FetchSession, Source
from src.fetch.rate_limit import FetchRateLimiter

logger = logger.bind(module="fetch assets")

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