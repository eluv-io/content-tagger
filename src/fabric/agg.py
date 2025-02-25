from typing import List, Dict, Tuple
import json
from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict
import os
from elv_client_py import ElvClient
import tempfile
from loguru import logger
from requests.exceptions import HTTPError

from common_ml.tags import AggTag
from common_ml.tags import VideoTag, FrameTag
from common_ml.utils import nested_update
from common_ml.utils.files import get_file_type, encode_path

from src.fabric.video import fetch_stream_metadata

def format_asset_tags(client: ElvClient, write_token: str) -> None:
    tmpdir = tempfile.TemporaryDirectory()
    save_path = tmpdir.name
    qlib = client.content_object_library_id(write_token=write_token)
    try:
        client.download_directory(dest_path=save_path, fabric_path=f"image_tags", write_token=write_token)
    except HTTPError as e:
        logger.error(f"Error downloading image tags: {e}")
        return
    file_to_tags = defaultdict(dict)
    for model in os.listdir(save_path):
        for tag in os.listdir(os.path.join(save_path, model)):
            with open(os.path.join(save_path, model, tag)) as f:
                tags = json.load(f)
            filename = tag.split("_imagetags.json")[0]
            trackname = label_to_track(feature_to_label(model))
            if "image_tags" not in file_to_tags[filename]:
                file_to_tags[filename]["image_tags"] = {}
            file_to_tags[filename]["image_tags"].update({trackname: {"tags": tags}})
    filetags = dict(file_to_tags)
    asset_metadata = client.content_object_metadata(write_token=write_token, metadata_subtree="assets", resolve_links=False)
    for asset, adata in asset_metadata.items():
        if not get_file_type(asset) == "image":
            continue
        filelink = adata.get("file", {}).get("/", None)
        if filelink is None or not filelink.startswith("./files"):
            logger.warning(f"Asset {asset} has no file link")
            continue
        filepath = filelink.split("./files/")[1]
        encoded = encode_path(filepath)
        if encoded not in filetags:
            logger.warning(f"No tags found for asset {asset}")
            continue
        asset_metadata[asset] = nested_update(adata, filetags[encoded])
        
    client.replace_metadata(write_token, asset_metadata, library_id=qlib, metadata_subtree="assets")

    tmpdir.cleanup()

def format_video_tags(client: ElvClient, write_token: str, streams: List[str], interval: int) -> None:
    """format_video_tags is used to format the tags for compatability with search and video editor.
    It operates on a content write token directly which should be published after the tags are uploaded. 
    
    Args:
        client: ElvClient object authenticated appropriately
        write_token: the write token for the content object we want to upload the formatted tags to
        streams: a list of tagged streams which we want to include in formatted results
        interval: the interval in minutes to bucket the formatted results (in minutes). (10 minutes is convention)
    """
    tmpdir = tempfile.TemporaryDirectory()
    save_path = tmpdir.name
    all_frame_tags, all_video_tags = {}, {}
    fps = None
    for stream in streams:
        try:
            client.download_directory(dest_path=os.path.join(save_path, write_token, stream), fabric_path=f"video_tags/{stream}", write_token=write_token)
        except HTTPError as e:
            logger.error(f"Error downloading video tags: {e}")
            return
        _, part_duration, fps, codec = fetch_stream_metadata(write_token, stream, client)
        for feature in os.listdir(os.path.join(save_path, write_token, stream)):
            if feature not in os.listdir(os.path.join(save_path, write_token, stream)):
                continue
            assert feature not in all_video_tags and feature not in all_frame_tags, f"Feature {feature} already found in another stream"
            tags_path = os.path.join(save_path, write_token, stream, feature)
            if codec == "video":
                frames_per_part = part_duration * fps
                if int(frames_per_part) != frames_per_part:
                    logger.warning("Calculated frames per part is not an integer, rounding down. This can be caused by variable FPS and may cause overlay tags to be misaligned.")
                frame_tags_files = [os.path.join(tags_path, file) for file in sorted(os.listdir(tags_path)) if file.endswith("_frametags.json")]
                if frame_tags_files:
                    all_frame_tags[feature] = merge_frame_tag_files(frame_tags_files, int(frames_per_part))
            video_tags_files = [os.path.join(tags_path, tag) for tag in sorted(os.listdir(tags_path)) if tag.endswith("_tags.json")]
            if len(video_tags_files) == 0:
                logger.warning(f"No tags found for feature {feature}")
                continue
            start_idx = int(os.path.basename(video_tags_files[0]).split("_")[0])
            offset = start_idx*part_duration
            all_video_tags[feature] = merge_video_tag_files(video_tags_files, part_duration, offset)

    assert "shot" in all_video_tags, "No shot tags found"
    intervals = [(tag.start_time, tag.end_time) for tag in all_video_tags["shot"]]
    agg_tags = aggregate_video_tags({f: tags for f, tags in all_video_tags.items() if f != "shot"}, intervals)
    formatted_tracks = format_tracks({"shot_tags": agg_tags}, all_video_tags, interval)
    overlays = format_overlay(all_frame_tags, fps, interval)

    to_upload = []
    for i, track in enumerate(formatted_tracks):
        fpath = os.path.join(save_path, write_token, f"video-tags-tracks-{i:04d}.json")
        to_upload.append(fpath)
        with open(fpath, 'w') as f:
            json.dump(track, f)

    for i, overlay in enumerate(overlays):
        fpath = os.path.join(save_path, write_token, f"video-tags-overlay-{i:04d}.json")
        to_upload.append(fpath)
        with open(fpath, 'w') as f:
            json.dump(overlay, f)
        
    jobs = [ElvClient.FileJob(local_path=path, out_path=f"video_tags/{os.path.basename(path)}", mime_type="application/json") for path in to_upload]
    qlib = client.content_object_library_id(write_token=write_token)
    client.upload_files(write_token=write_token, library_id=qlib, file_jobs=jobs)

    libid = client.content_object_library_id(write_token=write_token)
    for file in to_upload:
        basename = os.path.basename(file)
        add_link(client, basename, write_token, libid)

    tmpdir.cleanup()

def add_link(client: ElvClient, filename: str, qwt: str, libid: str) -> None:
    if 'video-tags-tracks' in filename:
        tag_type = 'metadata_tags'
    elif 'video-tags-overlay' in filename:
        tag_type = 'overlay_tags'
    else:
        return
    idx = ''.join([char for char in filename if char.isdigit()])

    data = {"/": f"./files/video_tags/{filename}"}
    client.merge_metadata(qwt, data, library_id=libid, metadata_subtree=f'video_tags/{tag_type}/{idx}')

def merge_video_tag_files(tags: List[str], tag_duration: float, offset: float) -> List[VideoTag]:
    tag_duration = tag_duration*1000
    offset = offset*1000
    merged = []
    for tag in tags:
        with open(tag, 'r') as f:
            data = json.load(f)
            data = [VideoTag(**tag) for tag in data]
        merged.extend([VideoTag(start_time=offset + tag.start_time, end_time=offset + tag.end_time, text=tag.text, confidence=tag.confidence) for tag in data])
        offset += tag_duration
    return merged

def merge_frame_tag_files(tags: List[str], len_frames: int) -> Dict[int, List[FrameTag]]:
    merged = {}
    frame_offset = 0
    for tag in tags:
        with open(tag, 'r') as f:
            data = json.load(f)
            data = {int(frame)+frame_offset: [FrameTag(**tag) for tag in tags] for frame, tags in data.items()}
        merged.update(data)
        frame_offset += len_frames
    return merged

def aggregate_video_tags(tags: Dict[str, List[VideoTag]], intervals: List[Tuple[int, int]]) -> List[AggTag]:
    all_tags = deepcopy(tags)
    for feature, tags in all_tags.items():
        all_tags[feature] = sorted(tags, key=lambda x: x.start_time)

    # merged tags into their appropriate intervals
    result = []
    for left, right in intervals:
        agg_tags = AggTag(start_time=left, end_time=right, tags={}) 
        for feature, tags in all_tags.items():
            for tag in tags:
                if tag.start_time >= left and tag.start_time < right:
                    if feature not in agg_tags.tags:
                        agg_tags.tags[feature] = []
                    agg_tags.tags[feature].append(tag)
        result.append(agg_tags)

    # TODO: not sure where else to define this custom logic
    for agg_tag in result:
        agg_tag.coalesce("asr")
    
    return result

def format_overlay(all_frame_tags: Dict[str, Dict[int, List[FrameTag]]], fps: float, interval: int) -> List[Dict[str, Dict[int, List[FrameTag]]]]:
    if len(all_frame_tags) == 0:
        return []
    buckets = defaultdict(lambda: {"version": 1, "overlay_tags": {"frame_level_tags": defaultdict(dict)}})
    interval = interval*1000*60 
    for feature, frame_tags in all_frame_tags.items():
        label = feature_to_label(feature)
        for frame_idx, ftags in frame_tags.items():
            timestamp_sec = frame_idx/fps
            bucket_idx = int(timestamp_sec/interval)
            buckets[bucket_idx]["overlay_tags"]["frame_level_tags"][frame_idx][label_to_track(label)] = {"tags": [asdict(tag) for tag in ftags]}
    buckets = [buckets[i] if i in buckets else {"version": 1, "overlay_tags": {"frame_level_tags": {}}} for i in range(max(buckets.keys())+1)]
    return buckets

# Args:
#   agg_tags: a dictionary mapping (label) -> list of aggregated tags (i.e shot_tags -> list of aggregated shot tags)
#   tracks: a dictionary mapping (feature name) -> list of tags (i.e the direct output of a 'service')
#   interval: formatted tracks are broken into buckets of this size (in minutes). If None, no bucketing is done.
#   
# Returns:
#    A tracks tag following the usual format for the video-tags-tracks files. Each element in the output list corresponds to one of the "video-tags-tracks-XXXX.json" files
def format_tracks(agg_tags: Dict[str, List[AggTag]], tracks: Dict[str, List[VideoTag]], interval: int) -> List[Dict[str, object]]:
    result = defaultdict(lambda: {"version": 1, "metadata_tags": {}})
    # convert to milliseconds
    interval = interval*1000*60 
    # add aggregated tags
    for key, tags in agg_tags.items():
        label = feature_to_label(key)
        for agg_tag in tags:
            entry = {
                "start_time": agg_tag.start_time,
                "end_time": agg_tag.end_time,
                "text": defaultdict(list)
            }
            bucket_idx = int(agg_tag.start_time/interval) if interval is not None else 0
            if key not in result[bucket_idx]["metadata_tags"]:
                result[bucket_idx]["metadata_tags"][key] = {"label": label, "tags": []}
            
            for track, video_tags in agg_tag.tags.items():
                track_label = feature_to_label(track)
                for vtag in video_tags:
                    as_dict = asdict(vtag)
                    if vtag.text is not None:
                        # NOTE: this is just a tag file convention, probably should just be a string value
                        as_dict["text"] = [as_dict["text"]]
                    entry["text"][track_label].append(as_dict) 
            result[bucket_idx]["metadata_tags"][key]["tags"].append(entry)

    # add standalone tracks
    for key, video_tags in tracks.items():
        label = feature_to_label(key)
        for vtag in video_tags:
            entry = {
                "start_time": vtag.start_time,
                "end_time": vtag.end_time,
            }
            if vtag.text is not None:
                entry["text"] = vtag.text
            bucket_idx = int(vtag.start_time/interval)
            if key not in result[bucket_idx]["metadata_tags"]:
                result[bucket_idx]["metadata_tags"][key] = {"label": label, "tags": []}
            result[bucket_idx]["metadata_tags"][key]["tags"].append(entry)

    # convert to list
    return [result[i] if i in result else {"version": 1, "metadata_tags": {}} for i in range(max(result.keys())+1)]

def feature_to_label(feature: str) -> str:
    if feature == "asr":
        return "Speech to Text"
    if feature == "caption":
        return "Object Detection"
    if feature == "celeb":
        return "Celebrity Detection"
    if feature == "logo":
        return "Logo Detection"
    if feature == "music":
        return "Music Detection"
    if feature == "ocr":
        return "Optical Character Recognition"
    if feature == "shot":
        return "Shot Detection"
    if feature == "llava":
        return "LLAVA Caption"
    return feature.title()

# e.g. "Shot Tags" -> "shot_tags"
def label_to_track(label: str) -> str:
    return label.lower().replace(" ", "_")