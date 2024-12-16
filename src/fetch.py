from elv_client_py import ElvClient
from requests.exceptions import HTTPError
import os
from loguru import logger
from typing import Optional, List
import threading

from common_ml.video_processing import unfrag_video
from config import config

class StreamNotFoundError(Exception):
    """Custom exception for specific error conditions."""
    pass

def fetch_stream(content_id: str, stream_name: str, output_path: str, client: ElvClient, start_time: Optional[int]=None, end_time: Optional[int]=None, replace: bool=False, exit_event: Optional[threading.Event]=None) -> List[str]:
    try:
        streams = client.content_object_metadata(object_id=content_id, metadata_subtree='offerings/default/media_struct/streams')
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve streams for content_id {content_id}") from e   
    if stream_name not in streams:
        raise StreamNotFoundError(f"Stream {stream_name} not found in content_id {content_id}")
    stream = streams[stream_name].get("sources", [])   
    if len(stream) == 0:
        raise StreamNotFoundError(f"Stream {stream_name} is empty")
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = float("inf")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    codec = streams[stream_name].get("codec_type", None)
    res = []
    for idx, part in enumerate(sorted(stream, key=lambda x: x["timeline_start"]["float"])):
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
            logger.info(f"Part hash:{part_hash} for content_id {content_id} already retrieved.")
            continue
        else:
            logger.info(f"Downloading part {part_hash} for content_id {content_id}")
        try:
            tmp_path = os.path.join(config["storage"]["tmp"], f"{idx}_{part_hash}")
            save_path = os.path.join(output_path, f"{idx}_{part_hash}.mp4")
            client.download_part(object_id=content_id, save_path=tmp_path, part_hash=part_hash)
            if codec == "video":
                unfrag_video(tmp_path, save_path)
            else:
                os.rename(tmp_path, save_path)
            res.append(save_path)
        except HTTPError as e:
            raise HTTPError(f"Failed to download part {part_hash} for content_id {content_id}") from e
    return res