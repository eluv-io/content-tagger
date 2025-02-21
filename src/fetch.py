from elv_client_py import ElvClient
from requests.exceptions import HTTPError
import os
from loguru import logger
from typing import Optional, List
import threading
import tempfile
import shutil
import subprocess

from common_ml.video_processing import unfrag_video
from common_ml.utils.files import get_file_type, encode_path
from common_ml.utils.metrics import timeit
from config import config

class StreamNotFoundError(Exception):
    """Custom exception for specific error conditions."""
    pass

def fetch_stream(content_id: str, stream_name: str, output_path: str, client: ElvClient, start_time: Optional[int]=None, end_time: Optional[int]=None, replace: bool=False, exit_event: Optional[threading.Event]=None) -> List[str]:
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = float("inf")
    
    try:
        transcodes = client.content_object_metadata(object_id=content_id, metadata_subtree='transcodes', resolve_links=False)
    except HTTPError:
        logger.warning(f"Transcodes not found for content_id {content_id}. Falling back to legacy method.")
        return fetch_stream_legacy(content_id, stream_name, output_path, client, start_time, end_time, replace, exit_event)
    
    try:
        streams = client.content_object_metadata(object_id=content_id, metadata_subtree='offerings/default/playout/streams')
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve streams for content_id {content_id}") from e
    
    if stream_name not in streams:
        raise StreamNotFoundError(f"Stream {stream_name} not found in content_id {content_id}")
    
    representations = streams[stream_name].get("representations", {})

    transcode_id = None
    for repr in representations.values():
        tid = repr["transcode_id"]
        if transcode_id is None:
            transcode_id = tid
            continue
        if tid != transcode_id:
            logger.error(f"Multiple transcode_ids found for stream {stream_name} in content_id {content_id}! Continuing with the first one found.")

    if transcode_id is None:
        raise StreamNotFoundError(f"Transcode_id not found for stream {stream_name} in content_id {content_id}")

    transcode_meta = transcodes[transcode_id]["stream"]
    codec_type = transcode_meta["codec_type"]
    stream = transcode_meta["sources"]

    return _download_stream(content_id, stream_name, output_path, client, codec_type, stream, start_time, end_time, replace, exit_event)

def fetch_stream_legacy(content_id: str, stream_name: str, output_path: str, client: ElvClient, start_time: Optional[int]=None, end_time: Optional[int]=None, replace: bool=False, exit_event: Optional[threading.Event]=None) -> List[str]:
    try:
        streams = client.content_object_metadata(object_id=content_id, metadata_subtree='offerings/default/media_struct/streams')
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve streams for content_id {content_id}") from e
    if stream_name not in streams:
        raise StreamNotFoundError(f"Stream {stream_name} not found in content_id {content_id}")
    stream = streams[stream_name].get("sources", [])
    if len(stream) == 0:
        raise StreamNotFoundError(f"Stream {stream_name} is empty")
    
    codec = streams[stream_name].get("codec_type", None)
    
    return _download_stream(content_id, stream_name, output_path, client, codec, stream, start_time, end_time, replace, exit_event)

def _download_stream(content_id: str, stream_name: str, output_path: str, client: ElvClient, codec_type: str, stream_meta: List[dict], start_time: Optional[int]=None, end_time: Optional[int]=None, replace: bool=False, exit_event: Optional[threading.Event]=None) -> List[str]:
    tmp_path = tempfile.mkdtemp(dir=config["storage"]["tmp"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    res = []
    for idx, part in enumerate(sorted(stream_meta, key=lambda x: x["timeline_start"]["float"])):
        if exit_event is not None and exit_event.is_set():
            logger.warning(f"Downloading of stream {stream_name} for content_id {content_id} stopped.")
            break
        idx = str(idx).zfill(4)
        part_hash = part["source"]
        pstart, pend = part["timeline_start"]["float"], part["timeline_end"]["float"]
        if not(start_time <= pstart < end_time) and not(start_time <= pend < end_time):
            continue
        elif not replace and os.path.exists(os.path.join(output_path, f"{idx}_{part_hash}.mp4")):
            res.append(os.path.join(output_path, f"{idx}_{part_hash}.mp4"))
            continue
        else:
            logger.info(f"Downloading part {part_hash} for content_id {content_id}")
        try:
            tmpfile = os.path.join(tmp_path, f"{idx}_{part_hash}")
            save_path = os.path.join(output_path, f"{idx}_{part_hash}.mp4")
            client.download_part(object_id=content_id, save_path=tmpfile, part_hash=part_hash)
            if codec_type == "video":
                unfrag_video(tmpfile, save_path)
            else:
                os.rename(tmpfile, save_path)
            res.append(save_path)
        except (HTTPError, RuntimeError) as e:
            if os.path.exists(save_path):
                # remove the corrupt file if it exists
                os.remove(save_path)
            logger.error(f"Failed to download part {part_hash} for content_id {content_id}: {str(e)}")
            continue
    shutil.rmtree(tmp_path, ignore_errors=True)
    return res

class AssetsNotFoundException(Exception):
    """Custom exception for specific error conditions."""
    pass

def fetch_assets(content_id: str, output_path: str, client: ElvClient, assets: Optional[List[str]], replace: bool=False) -> List[str]:
    if assets is None:
        try:
            assets_meta = client.content_object_metadata(object_id=content_id, metadata_subtree='assets')
            assets_meta = list(assets_meta.values())
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve assets for content_id {content_id}") from e
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
        raise AssetsNotFoundException(f"No image assets found in content_id {content_id}")
    logger.info(f"Found {len(assets)} image assets out of {total_assets} assets for content_id {content_id}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    to_download = []
    for asset in assets:
        asset_id = encode_path(asset)
        if replace or not os.path.exists(os.path.join(output_path, asset_id)):
            to_download.append(asset)
    if len(to_download) != len(set(to_download)):
        raise ValueError(f"Duplicate assets found for content_id {content_id}")
    if len(to_download) < len(assets):
        logger.info(f"{len(assets) - len(to_download)} assets already retrieved for {content_id}")
    logger.info(f"{len(to_download)} assets need to be downloaded for {content_id}")
    new_assets = _download_concurrent(client, to_download, content_id, output_path)
    bad_assets = set(to_download) - set(new_assets)
    assets = [asset for asset in assets if asset not in bad_assets]
    return [os.path.join(output_path, encode_path(asset)) for asset in assets]

def _download_sequential(client: ElvClient, files: List[str], exit_event: Optional[threading.Event], content_id: str, output_path: str) -> List[str]:
    res = []
    for asset in files:
        asset_id = encode_path(asset)
        if exit_event is not None and exit_event.is_set():
            logger.warning(f"Downloading of asset {asset} for content_id {content_id} stopped.")
            break
        try:
            save_path = os.path.join(output_path, asset_id)
            client.download_file(object_id=content_id, dest_path=save_path, file_path=asset)
            res.append(asset)
        except HTTPError:
            continue
    return res
        
def _download_concurrent(client: ElvClient, files: List[str], content_id: str, output_path: str) -> List[str]:
    file_jobs = [(asset, encode_path(asset)) for asset in files]
    with timeit("Downloading assets"):
        status = client.download_files(object_id=content_id, dest_path=output_path, file_jobs=file_jobs)
    return [asset for (asset, _), error in zip(file_jobs, status) if error is None]