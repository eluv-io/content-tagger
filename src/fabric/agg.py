from typing import List, Dict, Tuple, Optional
import json
from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict
import os
from loguru import logger
from requests.exceptions import HTTPError

from elv_client_py import ElvClient

from common_ml.tags import AggTag
from common_ml.tags import VideoTag, FrameTag
from common_ml.utils import nested_update
from common_ml.utils.files import get_file_type, encode_path

from src.fabric.video import fetch_stream_metadata
from src.fabric.utils import parse_qhit

def format_asset_tags(client: ElvClient, write_token: str, tags_path: str) -> None:
    qlib = client.content_object_library_id(write_token=write_token)
    image_tags_path = os.path.join(tags_path, 'image')
    try:
        res = _download_missing(client, image_tags_path, "image_tags", write_token=write_token)
    except HTTPError:
        logger.warning("No image tags")
        return

    for r in res:
        if r is not None:
            raise r
        
    file_to_tags = defaultdict(dict)
    for model in os.listdir(image_tags_path):
        for tag in os.listdir(os.path.join(image_tags_path, model)):
            with open(os.path.join(image_tags_path, model, tag)) as f:
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

def format_video_tags(client: ElvClient, write_token: str, interval: int, tags_path: str) -> None:
    """format_video_tags is used to format the tags for compatability with search and video editor.
    It operates on a content write token directly which should be published after the tags are uploaded. 
    
    Args:
        client: ElvClient object authenticated appropriately
        write_token: the write token for the content object we want to upload the formatted tags to
        streams: a list of tagged streams which we want to include in formatted results
        interval: the interval in minutes to bucket the formatted results (in minutes). (10 minutes is convention)
    """

    content_args = parse_qhit(write_token)
    qlib = client.content_object_library_id(**content_args)

    try:
        # get all tagged streams from fabric
        video_streams = client.list_files(qlib, path="/video_tags", **content_args)
        video_streams = [path.split("/")[0] for path in video_streams if path.endswith("/") and path[:-1] != "image"]
    except HTTPError:
        logger.debug("No tagged video streams found on fabric.")
        return
        
    all_frame_tags, all_video_tags = {}, {}
    custom_labels = {}
    
    if "source_tags" in video_streams:
        res = client.download_directory(dest_path=os.path.join(tags_path, 'source_tags'), fabric_path="video_tags/source_tags", write_token=write_token)
        for r in res:
            if r is not None:
                raise r
        logger.info("Parsing external tags")
        external_tags, labels = _parse_external_tags(os.path.join(tags_path, 'source_tags'))
        custom_labels.update(labels)
        all_video_tags.update(external_tags)
        video_streams.remove("source_tags")

    fps = None
    for stream in video_streams:
        stream_save_path = os.path.join(tags_path, stream)
        res = _download_missing(client, save_path=stream_save_path, fabric_path=f"video_tags/{stream}", write_token=write_token)
        for r in res:
            if r is not None:
                raise r

        stream_tracks = os.listdir(stream_save_path)
        logger.debug(f"stream_tracks for {stream}: {stream_tracks}")
        _, part_duration, fps, codec = fetch_stream_metadata(write_token, stream, client)
        for feature in os.listdir(stream_save_path):
            if feature not in os.listdir(stream_save_path):
                continue
            assert feature not in all_video_tags and feature not in all_frame_tags, f"Feature {feature} already found in another stream"
            tags_path = os.path.join(stream_save_path, feature)
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
            all_video_tags[feature] = merge_video_tag_files(video_tags_files, part_duration)

    assert "shot" in all_video_tags, "No shot tags found"
    shot_intervals = [(tag.start_time, tag.end_time) for tag in all_video_tags["shot"]]
    agg_tags = aggregate_video_tags({f: tags for f, tags in all_video_tags.items() if f != "shot"}, shot_intervals)
    if "asr" in all_video_tags:
        # sentence level aggregation on speech to text
        sentence_intervals = _get_sentence_intervals(all_video_tags["asr"])
        sentence_agg_tags = aggregate_video_tags({"asr": all_video_tags["asr"]}, sentence_intervals)
        stt_sent_track = [VideoTag(agg_tag.start_time, agg_tag.end_time, agg_tag.tags["asr"][0].text) for agg_tag in sentence_agg_tags if "asr" in agg_tag.tags]
        all_video_tags["auto_captions"] = stt_sent_track

    formatted_tracks = format_tracks({"shot_tags": agg_tags}, all_video_tags, interval, custom_labels=custom_labels)
    overlays = format_overlay(all_frame_tags, fps, interval)
    to_upload = []
    for i, track in enumerate(formatted_tracks):
        fpath = os.path.join(tags_path, f"video-tags-tracks-{i:04d}.json")
        to_upload.append(fpath)
        with open(fpath, 'w') as f:
            json.dump(track, f)

    for i, overlay in enumerate(overlays):
        fpath = os.path.join(tags_path, f"video-tags-overlay-{i:04d}.json")
        to_upload.append(fpath)
        with open(fpath, 'w') as f:
            json.dump(overlay, f)
        
    logger.debug(f"upload files {to_upload}")
    jobs = [ElvClient.FileJob(local_path=path, out_path=f"video_tags/{os.path.basename(path)}", mime_type="application/json") for path in to_upload]
    client.upload_files(write_token=write_token, library_id=qlib, file_jobs=jobs, finalize=False)

    logger.debug("done uploading files")

    libid = client.content_object_library_id(write_token=write_token)
    for file in to_upload:
        basename = os.path.basename(file)
        add_link(client, basename, write_token, libid)

    logger.debug("done linking files")

def _download_missing(client: ElvClient, save_path: str, fabric_path: str, write_token: str) -> List[Optional[ValueError]]:
    """Recursively downloads the given fabric path into save_path only if there is a difference"""
    file_info = client.list_files(write_token=write_token, path=fabric_path, get_info=True)
    to_download = []
    status = [0, 0, 0]

    def helper(data: dict, sub_path: str):
        for key, value in data.items():
            if key == ".":
                continue
            if "." in value and value["."].get("type", "") == "directory":
                helper(value, "/".join([sub_path, key]) if sub_path != "" else key)
            else:
                fpath = "/".join([sub_path, key]) if sub_path != "" else key
                fsize = value["."]["size"]
                if not os.path.exists(os.path.join(save_path, fpath)):
                    to_download.append(fpath)
                    status[0] += 1
                elif fsize != os.path.getsize(os.path.join(save_path, fpath)):
                    to_download.append(fpath)
                    status[1] += 1
                else:
                    status[2] += 1
                    
    helper(file_info, sub_path="")
    new, changed, old = status
    logger.debug(f"{new} new files found on fabric. {changed} have changed on fabric. {old} files already up to date")

    return client.download_files([("/".join([fabric_path, path]), path) for path in to_download], dest_path=save_path, write_token=write_token)

def _parse_external_tags(tags_path: str) -> Dict[str, List[VideoTag]]:
    external_tags, labels = {}, {}
    for tag_type in os.listdir(tags_path):
        for tag_file in os.listdir(os.path.join(tags_path, tag_type)):
            with open(os.path.join(tags_path, tag_type, tag_file), 'r') as f:
                data = json.load(f)["metadata_tags"]
            for feature in data:
                labels[feature] = data[feature]["label"]
                external_tags[feature] = _parse_external_track(data[feature])
    return external_tags, labels

def _get_sentence_intervals(tags: List[VideoTag]) -> List[Tuple[int, int]]:
    sentence_delimiters = ['.', '?', '!']
    intervals = []
    if len(tags) == 0:
        return []
    quiet = True
    curr_int = [0]
    for i, tag in enumerate(tags):
        assert tag.text is not None 
        if quiet and tag.start_time > curr_int[0]:
            # commit the silent interval
            curr_int.append(tag.start_time)
            intervals.append((curr_int[0], curr_int[-1]))
            curr_int.clear()
            # start a new speaking interval
            curr_int.append(tag.start_time)
            quiet = False
        if tag.text[-1] in sentence_delimiters or i == len(tags)-1:
            # end and commit the speaking interval, add one due to exclusive bounds
            curr_int.append(tag.end_time+1)
            intervals.append((curr_int[0], curr_int[-1]))
            curr_int.clear()
            # start a new silent interval
            curr_int.append(tag.end_time+1)
            quiet = True
    return intervals

def _parse_external_track(data: List[Dict[str, object]]) -> List[VideoTag]:
    track = []
    for tag in data["tags"]:
        track.append(VideoTag(start_time=tag["start_time"], end_time=tag["end_time"], text="; ".join(tag["text"])))
    return track

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

def merge_video_tag_files(tags: List[str], part_duration: float) -> List[VideoTag]:
    """Merges all the VideoTags from the given list of files into a single list of VideoTags with global timestamps."""
    tag_duration = part_duration*1000
    merged = []
    for tag in tags:
        part_idx = int(os.path.basename(tag).split("_")[0])
        part_start = part_idx * tag_duration
        with open(tag, 'r') as f:
            data = json.load(f)
            data = [VideoTag(**tag) for tag in data]
        merged.extend([VideoTag(start_time=part_start + tag.start_time, end_time=part_start + tag.end_time, text=tag.text, confidence=tag.confidence) for tag in data])
    return merged

def merge_frame_tag_files(tags: List[str], len_frames: int) -> Dict[int, List[FrameTag]]:
    merged = {}
    for tag in tags:
        part_idx = int(os.path.basename(tag).split("_")[0])
        part_start = part_idx * len_frames
        with open(tag, 'r') as f:
            data = json.load(f)
            data = {int(frame)+part_start: [FrameTag(**tag) for tag in tags] for frame, tags in data.items()}
        merged.update(data)
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
    interval = interval*60 
    for feature, frame_tags in all_frame_tags.items():
        label = feature_to_label(feature)
        for frame_idx, ftags in frame_tags.items():
            timestamp_sec = frame_idx/fps
            bucket_idx = int(timestamp_sec/interval)
            buckets[bucket_idx]["overlay_tags"]["frame_level_tags"][frame_idx][label_to_track(label)] = {"tags": [asdict(tag) for tag in ftags]}
    # add timestamps
    for bucket in buckets.values():
        for frame_idx in bucket["overlay_tags"]["frame_level_tags"]:
            bucket["overlay_tags"]["frame_level_tags"][frame_idx]["timestamp_sec"] = int((frame_idx/fps) * 1000)
    buckets = [buckets[i] if i in buckets else {"version": 1, "overlay_tags": {"frame_level_tags": {}}} for i in range(max(buckets.keys())+1)]
    return buckets

# Args:
#   agg_tags: a dictionary mapping (label) -> list of aggregated tags (i.e shot_tags -> list of aggregated shot tags)
#   tracks: a dictionary mapping (feature name) -> list of tags (i.e the direct output of a 'service')
#   interval: formatted tracks are broken into buckets of this size (in minutes). If None, no bucketing is done.
#   
# Returns:
#    A tracks tag following the usual format for the video-tags-tracks files. Each element in the output list corresponds to one of the "video-tags-tracks-XXXX.json" files
def format_tracks(agg_tags: Dict[str, List[AggTag]], tracks: Dict[str, List[VideoTag]], interval: int, custom_labels: Optional[Dict[str, str]]=None) -> List[Dict[str, object]]:
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
                if track in custom_labels:
                    track_label = custom_labels[track]
                else:
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
        if key in custom_labels:
            label = custom_labels[key]
        else:
            label = feature_to_label(key)
        for vtag in video_tags:
            entry = {
                "start_time": vtag.start_time,
                "end_time": vtag.end_time,
            }
            if vtag.text is not None:
                entry["text"] = vtag.text
            bucket_idx = int(vtag.start_time/interval)
            track = label_to_track(label)
            if track not in result[bucket_idx]["metadata_tags"]:
                result[bucket_idx]["metadata_tags"][track] = {"label": label, "tags": []}
            result[bucket_idx]["metadata_tags"][track]["tags"].append(entry)

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
    return feature.replace("_", " ").title()

# e.g. "Shot Tags" -> "shot_tags"
def label_to_track(label: str) -> str:
    return label.lower().replace(" ", "_")
