from dataclasses import dataclass, asdict
from collections import defaultdict
from loguru import logger

from src.tags.tagstore.tagstore import FilesystemTagStore
from common_ml.tags import FrameTag, VideoTag
from src.tags.tagstore.types import UploadJob, Tag
from src.tags.legacy.agg import (
    aggregate_video_tags, 
    format_tracks, 
    _get_sentence_intervals
)
from src.tags.legacy_format import *

@dataclass
class TagConverterConfig:
    interval: int
    name_mapping: dict[str, str]

@dataclass
class JobWithTags:
    job: UploadJob
    tags: list[Tag]

def get_latest_tags_for_content(qhit: str, ts: FilesystemTagStore) -> list[JobWithTags]:
    """Get tags from the latest job for each source for the given content."""
    job_ids = ts.find_jobs(qhit=qhit)
    if not job_ids:
        return []

    jobs = [ts.get_job(job_id) for job_id in job_ids]
    jobs = [job for job in jobs if job is not None]
    jobs.sort(key=lambda job: job.id, reverse=True)  # Newest first

    tags = []
    sources_tagged = set()

    # For each job (newest first), collect tags for sources we haven't seen
    for job in jobs:
        job_sources = set()
        new_tags = []
        
        for tag in ts.get_tags(job.id):
            if tag.source in sources_tagged:
                continue
            job_sources.add(tag.source)
            new_tags.append(tag)
        
        sources_tagged.update(job_sources)
        tags.append(JobWithTags(job=job, tags=new_tags))

    return tags

class TagConverter:
    """
    Convert tags from tagstore format to fabric format and upload them.
    """

    def __init__(self, cfg: TagConverterConfig):
        self.cfg = cfg

    def convert(self, job_tags: list[JobWithTags]) -> tuple[list[dict], list[dict]]:
        """Convert tagstore tags to VideoTag and FrameTag formats."""
        vtags, ftags = {}, {}
        
        for jt in job_tags:
            job = jt.job
            if not job:
                logger.warning(f"Could not find job {jt.job.id} for tag")   
                continue
                
            feature = job.track

            for tag in jt.tags:
                frame_info = tag.additional_info.get("frame_tags", {})
                
                # Initialize feature if not seen before
                if feature not in vtags:
                    vtags[feature] = []
                    if frame_info:
                        ftags[feature] = {}
                
                # Add video tag
                vtags[feature].append(VideoTag(tag.start_time, tag.end_time, tag.text))
                
                # Add frame tags if present - use timestamp directly as frame index
                if frame_info:
                    for timestamp_str, fdata in frame_info.items():
                        # timestamp_str should be the global timestamp in milliseconds
                        timestamp_ms = int(timestamp_str)
                        if timestamp_ms not in ftags[feature]:
                            ftags[feature][timestamp_ms] = []
                        ftags[feature][timestamp_ms].append(
                            FrameTag(tag.text, fdata["box"], confidence=fdata["confidence"])
                        )

        # Handle shot intervals for aggregation
        shot_intervals = []
        if "shot" in vtags:
            shot_intervals = [(tag.start_time, tag.end_time) for tag in vtags["shot"]]

        # Aggregate tags by shot intervals if available
        aggshot_tags = {}
        if shot_intervals:
            non_shot_tags = {f: tags for f, tags in vtags.items() if f != "shot"}
            aggshot_tags = {
                "shot_tags": aggregate_video_tags(non_shot_tags, shot_intervals)
            }

        # Handle automatic captions from ASR
        if "asr" in vtags:
            try:
                sentence_intervals = _get_sentence_intervals(vtags["asr"])
                sentence_agg_tags = aggregate_video_tags(
                    {"asr": vtags["asr"]}, 
                    sentence_intervals
                )
                stt_sent_track = [
                    VideoTag(agg_tag.start_time, agg_tag.end_time, agg_tag.tags["asr"][0].text) 
                    for agg_tag in sentence_agg_tags if "asr" in agg_tag.tags
                ]
                vtags["auto_captions"] = stt_sent_track
            except Exception as e:
                logger.warning(f"Failed to create auto captions from ASR: {e}")

        # Format tracks and overlays
        formatted_tracks = format_tracks(
            aggshot_tags, 
            vtags, 
            self.cfg.interval, 
            custom_labels={}
        )
        # Remove fps parameter since frame timestamps are already global
        overlays = self.format_overlay(ftags, self.cfg.interval)
        return formatted_tracks, overlays

    def format_overlay(self, all_frame_tags: dict[str, dict[int, list[FrameTag]]], interval: int) -> list[dict[str, dict[int, list[FrameTag]]]]:
        if len(all_frame_tags) == 0:
            return []
        buckets = defaultdict(lambda: {"version": 1, "overlay_tags": {"frame_level_tags": defaultdict(dict)}})
        interval = interval*60*1000  # Convert to milliseconds
        
        for feature, frame_tags in all_frame_tags.items():
            label = self.feature_to_label(feature)
            for frame_idx, ftags in frame_tags.items():
                timestamp_ms = frame_idx
                bucket_idx = int(frame_idx/interval)
                buckets[bucket_idx]["overlay_tags"]["frame_level_tags"][frame_idx][self.label_to_track(label)] = {"tags": [asdict(tag) for tag in ftags]}
        
        # Add timestamps (frame_idx is already in milliseconds)
        for bucket in buckets.values():
            for frame_idx in bucket["overlay_tags"]["frame_level_tags"]:
                bucket["overlay_tags"]["frame_level_tags"][frame_idx]["timestamp_sec"] = frame_idx
        
        buckets = [buckets[i] if i in buckets else {"version": 1, "overlay_tags": {"frame_level_tags": {}}} for i in range(max(buckets.keys())+1)]
        return buckets
    
    def feature_to_label(self, feature: str) -> str:
        if feature in self.cfg.name_mapping:
            return self.cfg.name_mapping[feature]
        return feature.replace("_", " ").title()

    def label_to_track(self, label: str) -> str:
        return label.lower().replace(" ", "_")