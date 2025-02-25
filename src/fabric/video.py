from typing import List, Optional, Tuple
from elv_client_py import ElvClient
from requests.exceptions import HTTPError
import os
import threading
import tempfile
import shutil
from common_ml.video_processing import unfrag_video
from common_ml.utils.files import get_file_type, encode_path
from common_ml.utils.metrics import timeit
from config import config
from loguru import logger

from .utils import parse_qhit

class StreamNotFoundError(RuntimeError):
    """Custom exception for specific error conditions."""
    pass

def fetch_stream(qhit: str, stream_name: str, output_path: str, client: ElvClient, start_time: Optional[int]=None, end_time: Optional[int]=None, replace: bool=False, exit_event: Optional[threading.Event]=None) -> List[str]:
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = float("inf")

    if _is_live(qhit, client):
        parts, codec_type = _fetch_livestream_parts(qhit, stream_name, client)
    elif _is_legacy_vod(qhit, client):
        parts, codec_type = _fetch_parts_vod_legacy(qhit, stream_name, client)
    else:
        parts, codec_type = _fetch_parts_vod(qhit, stream_name, client)
    
    return _download_parts(qhit, output_path, client, codec_type, parts, start_time, end_time, replace, exit_event)

def _fetch_parts_vod(qhit: str, stream_name: str, client: ElvClient) -> Tuple[List[tuple], str]:
    """Returns the parts with start/end time & the codec for the stream."""

    try:
        transcodes = client.content_object_metadata(metadata_subtree='transcodes', resolve_links=False, **parse_qhit(qhit))
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve transcodes for {qhit}") from e

    try:
        streams = client.content_object_metadata(metadata_subtree='offerings/default/playout/streams', resolve_links=False, **parse_qhit(qhit))
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve streams for {qhit}") from e
    
    if stream_name not in streams:
        raise StreamNotFoundError(f"Stream {stream_name} not found in {qhit}")
    
    representations = streams[stream_name].get("representations", {})

    transcode_id = None
    for repr in representations.values():
        tid = repr["transcode_id"]
        if transcode_id is None:
            transcode_id = tid
            continue
        if tid != transcode_id:
            logger.error(f"Multiple transcode_ids found for stream {stream_name} in {qhit}! Continuing with the first one found.")

    if transcode_id is None:
        raise StreamNotFoundError(f"Transcode_id not found for stream {stream_name} in {qhit}")

    transcode_meta = transcodes[transcode_id]["stream"]
    codec_type = transcode_meta["codec_type"]
    stream = transcode_meta["sources"]

    parts = [(part["source"], part["timeline_start"]["float"], part["timeline_end"]["float"]) for part in stream]

    return parts, codec_type

def _fetch_parts_vod_legacy(qhit: str, stream_name: str, client: ElvClient) -> Tuple[List[tuple], str]:
    """Returns the parts with start/end time & the codec for the stream."""

    try:
        streams = client.content_object_metadata(metadata_subtree='offerings/default/media_struct/streams', resolve_links=False, **parse_qhit(qhit))
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve streams for {qhit}") from e
    if stream_name not in streams:
        raise StreamNotFoundError(f"Stream {stream_name} not found in {qhit}")
    stream = streams[stream_name].get("sources", [])
    if len(stream) == 0:
        raise StreamNotFoundError(f"Stream {stream_name} is empty")
    
    parts = [(part["source"], part["timeline_start"]["float"], part["timeline_end"]["float"]) for part in stream]
    
    codec = streams[stream_name].get("codec_type", None)

    if codec is None:
        raise ValueError(f"Codec type not found for stream {stream_name} in {qhit}")
    
    return parts, codec

def _fetch_livestream_parts(qhit: str, stream_name: str, client: ElvClient) -> Tuple[List[tuple], str]:
    """Returns the livestream parts with start/end time & the codec for the stream."""

    try:
        periods = client.content_object_metadata(metadata_subtree='live_recording/recordings/live_offering', resolve_links=False, **parse_qhit(qhit))
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve periods for live recording {qhit}") from e
    if len(periods) == 0:
        raise StreamNotFoundError(f"Live recording {qhit} is empty")
    period = periods[-1]
    stream = period.get("sources", {}).get(stream_name, {}).get("parts", [])
    if len(stream) == 0:
        raise StreamNotFoundError(f"Stream {stream_name} not found in live recording {qhit}")
    if stream_name == "video":
        codec = "video"
    elif stream_name.startswith("audio"):
        codec = "audio"
    else:
        raise ValueError(f"Invalid stream name for live: {stream_name}. Must be 'video' or start with prefix 'audio'.")
    
    try:
        live_stream_info = client.content_object_metadata(metadata_subtree='live_recording/recording_config/recording_params/xc_params', resolve_links=False, **parse_qhit(qhit))
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve live stream metadata from {qhit}") from e

    if codec == "video":
        part_duration = live_stream_info.get("seg_duration", None)
        assert part_duration is not None, "Part duration not found in live stream metadata"
        part_duration = float(part_duration)
    else:
        sr = live_stream_info.get("sample_rate", None)
        ts = live_stream_info.get("audio_seg_duration_ts", None)
        assert sr is not None and ts is not None, "Sample rate or audio segment duration not found in live stream metadata"
        part_duration = int(ts) / int(sr)

    # filter out parts with close_time = 0, meaning the part is still live
    stream = [(part["part"], idx * part_duration, (idx+1)*part_duration) for idx, part in enumerate(stream) if part["close_time"] != 0]

    return stream, codec

def _download_parts(qhit: str, output_path: str, client: ElvClient, codec_type: str, parts: List[tuple], start_time: Optional[int]=None, end_time: Optional[int]=None, replace: bool=False, exit_event: Optional[threading.Event]=None) -> List[str]:
    """Downloads the parts from the stream.

    Args:
        parts (List[tuple]): List of tuples containing the part hash and the start and end times of the part: (part_hash, start_time, end_time)
        ...
    """
    
    tmp_path = tempfile.mkdtemp(dir=config["storage"]["tmp"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    res = []
    for idx, (part_hash, pstart, pend) in enumerate(sorted(parts, key=lambda x: x[1])):
        if exit_event is not None and exit_event.is_set():
            break
        idx = str(idx).zfill(4)
        if not(start_time <= pstart < end_time) and not(start_time <= pend < end_time):
            continue
        elif not replace and os.path.exists(os.path.join(output_path, f"{idx}_{part_hash}.mp4")):
            res.append(os.path.join(output_path, f"{idx}_{part_hash}.mp4"))
            continue
        else:
            logger.info(f"Downloading part {part_hash} for {qhit}")
        try:
            tmpfile = os.path.join(tmp_path, f"{idx}_{part_hash}")
            save_path = os.path.join(output_path, f"{idx}_{part_hash}.mp4")
            client.download_part(save_path=tmpfile, part_hash=part_hash, **parse_qhit(qhit))
            if codec_type == "video":
                unfrag_video(tmpfile, save_path)
            else:
                os.rename(tmpfile, save_path)
            res.append(save_path)
        except RuntimeError as e:
            if os.path.exists(save_path):
                # remove the corrupt file if it exists
                os.remove(save_path)
            logger.error(f"Failed to download part {part_hash} for {qhit}: {str(e)}")
            continue
    shutil.rmtree(tmp_path, ignore_errors=True)
    return res

def _is_live(qhit: str, client: ElvClient) -> bool:
    if not qhit.startswith("tqw__"):
        return False
    try:
        client.content_object_metadata(metadata_subtree='live_recording', write_token=qhit)
    except HTTPError:
        return False
    return True

def _is_legacy_vod(qhit: str, client: ElvClient) -> bool:
    try:
        client.content_object_metadata(metadata_subtree='transcodes', **parse_qhit(qhit))
    except HTTPError:
        return True
    return False