from typing import List, Dict, Tuple, Optional
import json
from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict
import os
from loguru import logger

from common_ml.tags import AggTag
from common_ml.tags import VideoTag, FrameTag

from src.tags.labels import feature_to_label, label_to_track

from config import config

def _parse_external_tags(tags_path: str) -> tuple[dict[str, list[VideoTag]], dict]:
    external_tags, labels = {}, {}
    for tag_type in os.listdir(tags_path):
        for tag_file in os.listdir(os.path.join(tags_path, tag_type)):
            logger.debug(f"external tags type: {tag_type} file: {tag_file}")
            try:
                with open(os.path.join(tags_path, tag_type, tag_file), 'r') as f:
                    data = json.load(f)["metadata_tags"]
            except Exception as e:
                logger.error(f"Error parsing external tags file {tag_file}: {e}")
                continue
            for feature in data:
                labels[feature] = data[feature]["label"]
                external_tags[feature] = _parse_external_track(data[feature])
    return external_tags, labels

MAX_SENTENCE_WORDS = 250
def _get_sentence_intervals(tags: List[VideoTag]) -> List[Tuple[int, int]]:
    sentence_delimiters = ['.', '?', '!']
    intervals = []
    if len(tags) == 0:
        return []
    quiet = True
    curr_int = [0]
    fake_sentence_cutoff = MAX_SENTENCE_WORDS
    for i, tag in enumerate(tags):
        if not tag.text:
            continue
        assert tag.text is not None
        if quiet and tag.start_time > curr_int[0]:
            # commit the silent interval
            curr_int.append(tag.start_time)
            intervals.append((curr_int[0], curr_int[-1]))
            curr_int.clear()
            # start a new speaking interval
            curr_int.append(tag.start_time)
            quiet = False
        if tag.text[-1] in sentence_delimiters or i == len(tags)-1 or i > fake_sentence_cutoff:
            fake_sentence_cutoff = i + MAX_SENTENCE_WORDS
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
        tagdata = tag["text"]
        if type(tagdata) == list:
            tagdata = "; ".join(tagdata)
        track.append(VideoTag(start_time=tag["start_time"], end_time=tag["end_time"], text=tagdata))
    return track

def merge_video_tag_files(tags: List[str], part_duration: float, combine_across_parts: bool = False) -> List[VideoTag]:
    """Merges all the VideoTags from the given list of files into a single list of VideoTags with global timestamps."""
    tag_duration = part_duration*1000
    merged = []
    for tag in tags:
        part_idx = int(os.path.basename(tag).split("_")[0])
        part_start = part_idx * tag_duration
        try:
            with open(tag, 'r') as f:
                data = json.load(f)
                data = [VideoTag(**tag) for tag in data]
        except json.decoder.JSONDecodeError as jd:
            logger.error(f"ERROR Decoding File {tag}: {jd}") 
            raise jd

        if combine_across_parts and len(merged) > 0 and merged[-1].end_time == part_start and len(data) > 0 and data[0].start_time == 0 and merged[-1].text == data[0].text:
            logger.debug(f"Merging part {part_idx} with previous part due to part boundary alignment")
            merged[-1].end_time = data[0].end_time + part_start
            data.pop(0)

        merged.extend([VideoTag(start_time=part_start + tag.start_time, end_time=part_start + tag.end_time, text=tag.text, confidence=tag.confidence) for tag in data])

    ## round the video tags, but do it after, so shots are more likely to get combined...
    return [ VideoTag(start_time=round(vt.start_time), end_time=round(vt.end_time), text=vt.text, confidence=vt.confidence) for vt in merged ]

def merge_frame_tag_files(tags: List[str], len_frames: int) -> Dict[int, List[FrameTag]]:
    merged = {}
    for tag in tags:
        part_idx = int(os.path.basename(tag).split("_")[0])
        part_start = part_idx * len_frames
        try:
            with open(tag, 'r') as f:
                data = json.load(f)
                data = {int(frame)+part_start: [FrameTag(**tag) for tag in tags] for frame, tags in data.items()}
        except json.decoder.JSONDecodeError as jd:
            logger.error(f"ERROR Decoding tag file {tag}: {jd}")
            raise jd
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
        for feat in config["agg"]["coalesce_features"]:
            agg_tag.coalesce(feat)

    for agg_tag in result:
        for feat in config["agg"]["single_shot_tag"]:
            agg_tag.keep_longest(feat)
    
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
            #label = custom_labels[key]
            continue # quick fix to not dupe the user tags
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
