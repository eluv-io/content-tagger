import os
from loguru import logger
import threading
from requests.exceptions import HTTPError

from common_ml.utils.files import get_file_type, encode_path
from common_ml.utils.metrics import timeit

from src.common.content import Content

class AssetsNotFoundException(Exception):
    pass

def fetch_assets(q: Content, output_path: str, assets: list[str] | None, replace: bool=False) -> tuple[list[str], list[str]]:
    if assets is None:
        try:
            assets_meta = q.content_object_metadata(metadata_subtree='assets')
            assets_meta = list(assets_meta.values())
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve assets for {q.qhit}") from e
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
        raise AssetsNotFoundException(f"No image assets found in {q.qhit}")
    logger.info(f"Found {len(assets)} image assets out of {total_assets} assets for {q.qhit}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    to_download = []
    for asset in assets:
        asset_id = encode_path(asset)
        if replace or not os.path.exists(os.path.join(output_path, asset_id)):
            to_download.append(asset)
    if len(to_download) != len(set(to_download)):
        raise ValueError(f"Duplicate assets found for {q.qhit}")
    if len(to_download) < len(assets):
        logger.info(f"{len(assets) - len(to_download)} assets already retrieved for {q.qhit}")
    logger.info(f"{len(to_download)} assets need to be downloaded for {q.qhit}")
    new_assets = _download_concurrent(q, to_download, output_path)
    bad_assets = set(to_download) - set(new_assets)
    assets = [asset for asset in assets if asset not in bad_assets]
    return [os.path.join(output_path, encode_path(asset)) for asset in assets], list(bad_assets)

def _download_sequential(q: Content, files: list[str], exit_event: threading.Event | None, output_path: str) -> list[str]:
    res = []
    for asset in files:
        asset_id = encode_path(asset)
        if exit_event is not None and exit_event.is_set():
            logger.warning(f"Downloading of asset {asset} for {q.qhit} stopped.")
            break
        try:
            save_path = os.path.join(output_path, asset_id)
            q.download_file(dest_path=save_path, file_path=asset)
            res.append(asset)
        except HTTPError:
            continue
    return res
        
def _download_concurrent(q: Content, files: list[str], output_path: str) -> list[str]:
    file_jobs = [(asset, encode_path(asset)) for asset in files]
    with timeit("Downloading assets"):
        status = q.download_files(dest_path=output_path, file_jobs=file_jobs)
    return [asset for (asset, _), error in zip(file_jobs, status) if error is None]