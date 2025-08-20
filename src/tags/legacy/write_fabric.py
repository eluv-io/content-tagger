import os
import json
from collections import defaultdict
from typing import List
from requests import HTTPError
from elv_client_py import ElvClient
from src.common.content import Content
from loguru import logger

from src.fabric.fetch_video import fetch_stream_metadata
from common_ml.tags import VideoTag
from common_ml.utils import nested_update
from common_ml.utils.metrics import timeit
from common_ml.utils.files import get_file_type, encode_path
from src.tags.legacy.agg import (
    aggregate_video_tags, 
    format_tracks, format_overlay, 
    merge_frame_tag_files, 
    merge_video_tag_files, 
    _parse_external_tags, 
    _get_sentence_intervals
)
from src.tags.legacy.fetch_tags import _download_missing
from src.tags.legacy.labels import label_to_track, feature_to_label


def format_asset_tags(q: Content, tags_path: str) -> None:
    image_tags_path = os.path.join(tags_path, 'image')
    try:
        res = _download_missing(q, image_tags_path, "image_tags")
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
    asset_metadata = q.content_object_metadata(metadata_subtree="assets", resolve_links=False)
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
        
    q.replace_metadata(metadata=asset_metadata, metadata_subtree="assets")

def format_video_tags(q: Content, interval: int, tags_path: str) -> None:
    """format_video_tags is used to format the tags for compatability with search and video editor.
    It operates on a content write token directly which should be published after the tags are uploaded. 
    
    Args:
        q: content object with write token
        streams: a list of tagged streams which we want to include in formatted results
        interval: the interval in minutes to bucket the formatted results (in minutes). (10 minutes is convention)
    """

    try:
        # get all tagged streams from fabric
        video_streams = q.list_files(path="/video_tags")
        video_streams = [path.split("/")[0] for path in video_streams if path.endswith("/") and path[:-1] != "image"]
    except HTTPError:
        video_streams = []

    if len(video_streams) == 0:
        logger.warning("No video tags found")
        return

    all_frame_tags, all_video_tags = {}, {}
    custom_labels = {}

    if "source_tags" in video_streams:
        with timeit("Downloading source tags"):
            res = q.download_directory(dest_path=os.path.join(tags_path, 'source_tags'), fabric_path="video_tags/source_tags")
        for r in res:
            if r is not None:
                raise r
        logger.info("Parsing external tags")
        external_tags, labels = _parse_external_tags(os.path.join(tags_path, 'source_tags'))
        custom_labels.update(labels)
        ##all_video_tags.update(external_tags)
        video_streams.remove("source_tags")
    else:
        external_tags = {}

    fps = None
    for stream in video_streams:
        stream_save_path = os.path.join(tags_path, stream)
        with timeit(f"Downloading tags for {stream}"):
            res = _download_missing(q, save_path=stream_save_path, fabric_path=f"video_tags/{stream}")
        for r in res:
            if r is not None:
                raise r

        stream_tracks = os.listdir(stream_save_path)
        logger.debug(f"stream_tracks for {stream}: {stream_tracks}")
        _, part_duration, fps, codec = fetch_stream_metadata(q, stream)
        for feature in os.listdir(stream_save_path):
            if feature not in os.listdir(stream_save_path):
                continue
            assert feature not in all_video_tags and feature not in all_frame_tags, f"Feature {feature} already found in another stream"
            assert feature not in external_tags, f"Feature {feature} already found in another stream in external tags"
            model_path = os.path.join(stream_save_path, feature)
            if codec == "video":
                frames_per_part = part_duration * fps
                if int(frames_per_part) != frames_per_part:
                    logger.warning("Calculated frames per part is not an integer, rounding down. This can be caused by variable FPS and may cause overlay tags to be misaligned.")
                frame_tags_files = [os.path.join(model_path, file) for file in sorted(os.listdir(model_path)) if file.endswith("_frametags.json")]
                if frame_tags_files:
                    all_frame_tags[feature] = merge_frame_tag_files(frame_tags_files, int(frames_per_part))
            video_tags_files = [os.path.join(model_path, tag) for tag in sorted(os.listdir(model_path)) if tag.endswith("_tags.json")]
            if len(video_tags_files) == 0:
                logger.warning(f"No tags found for feature {feature}")
                continue
            all_video_tags[feature] = merge_video_tag_files(video_tags_files, part_duration, combine_across_parts = False)#combine_across_parts = (feature == "shot"))

    ## shot must be entirely ml-tagger or entirely external tags, both is not allowed
    if "shot" in all_video_tags:
        shot_intervals = [(tag.start_time, tag.end_time) for tag in all_video_tags["shot"]]
    elif "shot" in external_tags:
        ## this is only true for the case of converting v1 shot tags into external tags (rare)
        shot_intervals = [(tag.start_time, tag.end_time) for tag in external_tags["shot"]]
    else:
        shot_intervals = []
    
    if len(shot_intervals) > 0:
        aggshot_tags = {"shot_tags": aggregate_video_tags({f: tags for f, tags in (list(all_video_tags.items()) + list(external_tags.items())) if f != "shot"}, shot_intervals) }
    else:
        aggshot_tags = {}

    if "asr" in all_video_tags:
        # sentence level aggregation on speech to text
        sentence_intervals = _get_sentence_intervals(all_video_tags["asr"])
        sentence_agg_tags = aggregate_video_tags({"asr": all_video_tags["asr"]}, sentence_intervals)
        stt_sent_track = [VideoTag(agg_tag.start_time, agg_tag.end_time, agg_tag.tags["asr"][0].text) for agg_tag in sentence_agg_tags if "asr" in agg_tag.tags]
        all_video_tags["auto_captions"] = stt_sent_track

    formatted_tracks = format_tracks(aggshot_tags, all_video_tags, interval, custom_labels=custom_labels)
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

    jobs = [ElvClient.FileJob(local_path=path, out_path=f"video_tags/{os.path.basename(path)}", mime_type="application/json") for path in to_upload]

    with timeit("Uploading aggregated files"):
        q.upload_files(file_jobs=jobs, finalize=False)

    logger.debug("done uploading files")

    with timeit("Adding links"):
        add_links(q, to_upload)

def add_link(q: Content, filename: str) -> None:
    if 'video-tags-tracks' in filename:
        tag_type = 'metadata_tags'
    elif 'video-tags-overlay' in filename:
        tag_type = 'overlay_tags'
    else:
        return
    idx = ''.join([char for char in filename if char.isdigit()])

    data = {"/": f"./files/video_tags/{filename}"}
    q.merge_metadata(metadata=data, metadata_subtree=f'video_tags/{tag_type}/{idx}')

def add_links(q: Content, fpaths: List[str]) -> None:
    data = {}
    for fpath in fpaths:
        filename = os.path.basename(fpath)
        if 'video-tags-tracks' in filename:
            tag_type = 'metadata_tags'
        elif 'video-tags-overlay' in filename:
            tag_type = 'overlay_tags'
        else:
            continue
        if tag_type not in data:
            data[tag_type] = {}
            
        idx = ''.join([char for char in filename if char.isdigit()])
        data[tag_type][idx] = {"/": f"./files/video_tags/{filename}"}
        
    q.merge_metadata(metadata=data, metadata_subtree='video_tags')