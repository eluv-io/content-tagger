import os
from loguru import logger
from typing import Optional, List
import threading
from requests.exceptions import HTTPError

from elv_client_py import ElvClient

from .utils import parse_qhit

from common_ml.utils.files import get_file_type, encode_path
from common_ml.utils.metrics import timeit

class AssetsNotFoundException(RuntimeError):
    """Custom exception for specific error conditions."""
    pass

def fetch_assets(qhit: str, output_path: str, client: ElvClient, assets: Optional[List[str]], replace: bool=False) -> List[str]:
    if assets is None:
        try:
            assets_meta = client.content_object_metadata(metadata_subtree='assets', **parse_qhit(qhit))
            assets_meta = list(assets_meta.values())
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve assets for {qhit}") from e
        assets = []
        for ameta in assets_meta:
            filepath = ameta.get("file")["/"]
            assert filepath.startswith("./files/")
            # strip leading term
            filepath = filepath[8:]
            assets.append(filepath)
    total_assets = len(assets)
    assets = [asset for asset in assets if get_file_type(asset) == "image"]
    if len(assets) == 0:
        raise AssetsNotFoundException(f"No image assets found in {qhit}")
    logger.info(f"Found {len(assets)} image assets out of {total_assets} assets for {qhit}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    to_download = []
    for asset in assets:
        asset_id = encode_path(asset)
        if replace or not os.path.exists(os.path.join(output_path, asset_id)):
            to_download.append(asset)
    if len(to_download) != len(set(to_download)):
        raise ValueError(f"Duplicate assets found for {qhit}")
    if len(to_download) < len(assets):
        logger.info(f"{len(assets) - len(to_download)} assets already retrieved for {qhit}")
    logger.info(f"{len(to_download)} assets need to be downloaded for {qhit}")
    new_assets = _download_concurrent(client, to_download, qhit, output_path)
    bad_assets = set(to_download) - set(new_assets)
    assets = [asset for asset in assets if asset not in bad_assets]
    return [os.path.join(output_path, encode_path(asset)) for asset in assets]

def _download_sequential(client: ElvClient, files: List[str], exit_event: Optional[threading.Event], qhit: str, output_path: str) -> List[str]:
    res = []
    for asset in files:
        asset_id = encode_path(asset)
        if exit_event is not None and exit_event.is_set():
            logger.warning(f"Downloading of asset {asset} for {qhit} stopped.")
            break
        try:
            save_path = os.path.join(output_path, asset_id)
            client.download_file(dest_path=save_path, file_path=asset, **parse_qhit(qhit))
            res.append(asset)
        except HTTPError:
            continue
    return res
        
def _download_concurrent(client: ElvClient, files: List[str], qhit: str, output_path: str) -> List[str]:
    file_jobs = [(asset, encode_path(asset)) for asset in files]
    with timeit("Downloading assets"):
        status = client.download_files(dest_path=output_path, file_jobs=file_jobs, **parse_qhit(qhit))
    return [asset for (asset, _), error in zip(file_jobs, status) if error is None]