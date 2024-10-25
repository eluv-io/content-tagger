from elv_client_py import ElvClient
from requests.exceptions import HTTPError
import os
from loguru import logger
from typing import Optional, List

class StreamNotFoundError(Exception):
    """Custom exception for specific error conditions."""
    pass

def fetch_stream(content_id: str, stream_name: str, output_path: str, client: ElvClient, start_time: Optional[int]=None, end_time: Optional[int]=None, replace: bool=False) -> List[str]:
    try:
        streams = client.content_object_metadata(object_id=content_id, metadata_subtree='offerings/default/media_struct/streams')
    except HTTPError as e:
        raise Exception(f"Failed to fetch streams for content_id {content_id}: {e}")
    if stream_name not in streams:
        raise StreamNotFoundError(f"Stream {stream_name} not found in content_id {content_id}")
    stream = streams[stream_name].get("sources", [])   
    if len(stream) == 0:
        raise Exception(f"Stream {stream_name} is empty")
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = float("inf")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    res = []
    for idx, part in enumerate(sorted(stream, key=lambda x: x["timeline_start"]["float"])):
        idx = str(idx).zfill(4)
        part_hash = part["source"]
        pstart, pend = part["timeline_start"]["float"], part["timeline_end"]["float"]
        if not(start_time <= pstart < end_time) and not(start_time <= pend < end_time):
            continue
        elif not replace and os.path.exists(os.path.join(output_path, f"{idx}_{part_hash}")):
            logger.info(f"Part hash:{part_hash} for content_id {content_id} already retrieved.")
        else:
            logger.info(f"Downloading part {part_hash} for content_id {content_id}")
        try:
            save_path = os.path.join(output_path, f"{idx}_{part_hash}")
            client.download_part(object_id=content_id, save_path=save_path, part_hash=part_hash)
            res.append(save_path)
        except HTTPError as e:
            raise Exception(f"Failed to download {part} for content_id {content_id}: {e}")
    return res