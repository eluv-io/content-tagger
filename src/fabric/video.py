from typing import Optional, Tuple
from requests.exceptions import HTTPError
import os
import threading
import tempfile
import shutil
from common_ml.video_processing import unfrag_video
from config import config
from loguru import logger

from src.fabric.content import Content
from src.api.errors import MissingResourceError, BadRequestError

class StreamNotFoundError(RuntimeError):
    """Custom exception for specific error conditions."""
    pass

def download_stream(
        q: Content,
        stream_name: str, 
        output_path: str, 
        start_time: Optional[int]=None,
        end_time: Optional[int]=None, 
        replace: bool=False, 
        exit_event: Optional[threading.Event]=None
    ) -> list[str]:
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = float("inf")

    parts, part_duration, _, codec_type = fetch_stream_metadata(q, stream_name)
    return _download_parts(q, output_path, codec_type, parts, part_duration, start_time, end_time, replace, exit_event)

def fetch_stream_metadata(q: Content, stream_name: str) -> Tuple[list[str], float, float | None, str]:
    if _is_live(q):
        return _fetch_livestream_metadata(q, stream_name)
    elif _is_legacy_vod(q):
        return _fetch_legacy_vod_metadata(q, stream_name)
    else:
        return _fetch_vod_metadata(q, stream_name)

def _fetch_vod_metadata(
        q: Content, 
        stream_name: str
    ) -> Tuple[list[str], float, float | None, str]:
    try:
        transcodes = q.content_object_metadata(metadata_subtree='transcodes', resolve_links=False)
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve transcodes for {q.qhit}") from e

    try:
        streams = q.content_object_metadata(metadata_subtree='offerings/default/playout/streams', resolve_links=False)
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve streams for {q.qhit}") from e
    
    if stream_name not in streams:
        raise MissingResourceError(f"Stream {stream_name} not found in {q.qhit}")
    
    representations = streams[stream_name].get("representations", {})

    transcode_id = None
    for repr in representations.values():
        tid = repr["transcode_id"]
        if transcode_id is None:
            transcode_id = tid
            continue
        if tid != transcode_id:
            logger.error(f"Multiple transcode_ids found for stream {stream_name} in {q.qhit}! Continuing with the first one found.")

    if transcode_id is None:
        raise MissingResourceError(f"Transcode_id not found for stream {stream_name} in {q.qhit}")

    transcode_meta = transcodes[transcode_id]["stream"]
    codec_type = transcode_meta["codec_type"]
    stream = transcode_meta["sources"]

    if len(stream) == 0:
        raise MissingResourceError(f"Stream {stream_name} is empty")
    
    part_duration = stream[0]["duration"]["float"]

    fps = None
    if codec_type == "video":
        fps = _parse_fps(transcode_meta["rate"])

    parts = [part["source"] for part in stream]

    return parts, part_duration, fps, codec_type

def _fetch_legacy_vod_metadata(
        q: Content, 
        stream_name: str
    ) -> tuple[list[str], float, float | None, str]:
    """Returns the parts with start/end time & the codec for the stream."""

    try:
        streams = q.content_object_metadata(metadata_subtree='offerings/default/media_struct/streams', resolve_links=False)
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve streams for {q.qhit}") from e
    
    if stream_name not in streams:
        raise MissingResourceError(f"Stream {stream_name} not found in {q.qhit}")
    
    stream = streams[stream_name].get("sources", [])
    if len(stream) == 0:
        raise MissingResourceError(f"Stream {stream_name} is empty")
    
    parts = [part["source"] for part in stream]
    
    codec = streams[stream_name].get("codec_type", None)
    part_duration = stream[0]["duration"]["float"]

    if codec is None:
        raise MissingResourceError(f"Codec type not found for stream {stream_name} in {q.qhit}")
    
    fps = None
    if codec == "video":
        fps = _parse_fps(streams[stream_name]["rate"])
    
    return parts, part_duration, fps, codec

def _fetch_livestream_metadata(q: Content, stream_name: str) -> tuple[list[str], float, float | None, str]:
    """Returns the livestream parts with start/end time & the codec for the stream."""

    try:
        periods = q.content_object_metadata(metadata_subtree='live_recording/recordings/live_offering', resolve_links=False)
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve periods for live recording {q.qhit}") from e
    if len(periods) == 0:
        raise MissingResourceError(f"Live recording {q.qhit} is empty")
    #if len(periods) > 1:
    #    raise StreamNotFoundError(f"Multiple periods found for live recording {qhit}. Multi-period tagging is not currently supported.")
    stream = periods[0].get("sources", {}).get(stream_name, {}).get("parts", [])
    if len(stream) == 0:
        raise MissingResourceError(f"Stream {stream_name} was found in live recording, but no parts were found.")
    if stream_name == "video":
        codec = "video"
    elif stream_name.startswith("audio"):
        codec = "audio"
    else:
        raise BadRequestError(f"Invalid stream name for live: {stream_name}. Must be 'video' or start with prefix 'audio'.")
    
    try:
        xc_params = q.content_object_metadata(metadata_subtree='live_recording/recording_config/recording_params/xc_params', resolve_links=False)
    except HTTPError as e:
        raise HTTPError(f"Failed to retrieve live stream metadata from {q.qhit}") from e
    
    if codec == "video":
        try:
            live_stream_info = q.content_object_metadata(metadata_subtree='live_recording_config/probe_info/streams', resolve_links=False)
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve live stream metadata from {q.qhit}") from e
        video_stream_info = None
        for stream_info in live_stream_info:
            if stream_info["codec_type"] == "video":
                video_stream_info = stream_info
                break
        assert video_stream_info is not None, "Video stream not found in live stream metadata"
        fps = _parse_fps(video_stream_info["frame_rate"])
        part_duration = xc_params.get("seg_duration", None)
        assert part_duration is not None, "Part duration not found in live stream metadata"
        part_duration = float(part_duration)
    else:
        sr = xc_params.get("sample_rate", None)
        ts = xc_params.get("audio_seg_duration_ts", None)
        assert sr is not None and ts is not None, "Sample rate or audio segment duration not found in live stream metadata"
        part_duration = int(ts) / int(sr)
        fps = None

    # filter out parts with finalization_time == 0, meaning the part is still live.
    stream = [part["hash"] for part in stream if part["finalization_time"] != 0 and part["size"] > 0]

    return stream, part_duration, fps, codec

def _download_parts(
        q: Content, 
        output_path: str, 
        codec_type: str, 
        parts: list[str], 
        part_duration: float, 
        start_time: int, 
        end_time: int, 
        replace: bool=False, 
        exit_event: Optional[threading.Event]=None
    ) -> tuple[list[str], list[str]]:
    """Downloads the parts from the stream."""
    
    tmp_path = tempfile.mkdtemp(dir=config["storage"]["tmp"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    res, failed = [], []
    for idx, part_hash in enumerate(parts):
        if exit_event is not None and exit_event.is_set():
            break
        pstart = idx * part_duration
        pend = (idx + 1) * part_duration
        idx = str(idx).zfill(4)
        if not(start_time <= pstart < end_time) and not(start_time <= pend < end_time):
            continue
        elif not replace and os.path.exists(os.path.join(output_path, f"{idx}_{part_hash}.mp4")):
            res.append(os.path.join(output_path, f"{idx}_{part_hash}.mp4"))
            continue
        else:
            logger.info(f"Downloading part {part_hash} for {q.qhit}")
        try:
            tmpfile = os.path.join(tmp_path, f"{idx}_{part_hash}")
            save_path = os.path.join(output_path, f"{idx}_{part_hash}.mp4")
            q.download_part(save_path=tmpfile, part_hash=part_hash)
            if codec_type == "video":
                unfrag_video(tmpfile, save_path)
            else:
                shutil.move(tmpfile, save_path)
            res.append(save_path)
        except Exception as e:
            if os.path.exists(save_path):
                # remove the corrupt file if it exists
                os.remove(save_path)
            failed.append(part_hash)
            logger.error(f"Failed to download part {part_hash} for {q.qhit}: {str(e)}")
            continue
    shutil.rmtree(tmp_path, ignore_errors=True)
    return res, failed

def _is_live(q: Content) -> bool:
    if not q.qhit.startswith("tqw__"):
        return False
    try:
        # TODO: make sure works for write token 
        q.content_object_metadata(metadata_subtree='live_recording')
    except HTTPError:
        return False
    return True

def _is_legacy_vod(q: Content) -> bool:
    try:
        q.content_object_metadata(metadata_subtree='transcodes')
    except HTTPError:
        return True
    return False

def _parse_fps(rat: str) -> float:
    if "/" in rat:
        num, den = rat.split("/")
        return float(num) / float(den)
    return float(rat)