from dataclasses import dataclass, asdict
from collections import defaultdict
from copy import deepcopy

from src.common.content import Content
from src.tags.tagstore.abstract import Tagstore
from common_ml.tags import FrameTag, VideoTag
from src.tags.tagstore.types import UploadJob, Tag
from src.tags.legacy_format import *

@dataclass
class TagConverterConfig:
    interval: int
    name_mapping: dict[str, str]
    coalesce_tracks: list[str]
    single_tag_tracks: list[str]
    max_sentence_words: int

@dataclass
class JobWithTags:
    job: UploadJob
    tags: list[Tag]
    
def get_latest_tags_for_content(q: Content, ts: Tagstore) -> list[JobWithTags]:
    """Get tags from the latest job for each source+track pair for the given content."""

    job_ids = ts.find_jobs(qhit=q.qid, q=q)
    if not job_ids:
        return []

    jobs = [ts.get_job(job_id, q=q) for job_id in job_ids]
    jobs = [job for job in jobs if job is not None]
    jobs.sort(key=lambda job: job.id, reverse=True)  # Newest first

    tags = []
    source_features_tagged = set()

    # For each job (newest first), collect tags for sources we haven't seen
    for job in jobs:
        new_tags = []

        for tag in ts.find_tags(jobid=job.id, q=q):
            if (tag.source, job.track) in source_features_tagged:
                continue
            new_tags.append(tag)

        source_features_tagged.update((tag.source, job.track) for tag in new_tags)

        tags.append(JobWithTags(job=job, tags=new_tags))

    return tags

class TagConverter:
    """
    Convert tags from tagstore format to fabric format and upload them.
    """

    def __init__(self, cfg: TagConverterConfig):
        self.cfg = cfg

    def split_tags(self, job_tags: list[JobWithTags]) -> list[list[JobWithTags]]:
        """Split based on start time"""
        num_buckets = int((max((tag.start_time for jt in job_tags for tag in jt.tags), default=0) / (self.cfg.interval*60*1000)) + 1)
        buckets: list[list[JobWithTags]] = [[] for _ in range(num_buckets)]
        for jt in job_tags:
            if not jt.tags:
                continue
            for bucket in buckets:
                bucket.append(JobWithTags(job=jt.job, tags=[]))
            for tag in jt.tags:
                bucket_idx = int(tag.start_time / (self.cfg.interval*60*1000))
                buckets[bucket_idx][-1].tags.append(tag)
        return buckets

    def get_tracks(self, job_tags: list[JobWithTags]) -> TrackCollection:
        vtags: dict[str, list[VideoTag]] = {}

        for jt in job_tags:
            job = jt.job

            feature = job.track

            for tag in jt.tags:
                
                if feature not in vtags:
                    vtags[feature] = []
                
                vtags[feature].append(VideoTag(tag.start_time, tag.end_time, tag.text))

        shot_jobs = [jt for jt in job_tags if jt.job.track == "shot"]
        assert len(shot_jobs) <= 1, "Multiple shot detection jobs found"

        shot_intervals = self._get_shot_intervals(shot_jobs[0].tags if shot_jobs else [])

        # Aggregate tags by shot intervals if available
        aggshot_tags = {}
        if shot_intervals:
            non_shot_tags = {f: tags for f, tags in vtags.items() if f != "shot"}
            aggshot_tags = {
                "shot_tags": self._aggregate_video_tags(non_shot_tags, shot_intervals)
            }

        # Handle automatic captions from ASR
        sentence_intervals = []
        if "asr" in vtags:
            sentence_intervals = self._get_sentence_intervals(vtags["asr"])

        aggsentence_tags = {}
        if sentence_intervals:
            aggsentence_tags = self._aggregate_video_tags({"asr": vtags["asr"]}, sentence_intervals)

        if aggsentence_tags:
            stt_sent_track = [
                VideoTag(agg_tag.start_time, agg_tag.end_time, agg_tag.tags["asr"][0].text)
                for agg_tag in aggsentence_tags if "asr" in agg_tag.tags
            ]
            vtags["auto_captions"] = stt_sent_track

        return TrackCollection(tracks=vtags, agg_tracks=aggshot_tags)

    def get_overlays(self, job_tags: list[JobWithTags]) -> Overlay:
        frame_tags: dict[int, dict[str, list[FrameTag]]] = {}

        for jt in job_tags:
            job = jt.job
            feature = job.track

            for tag in jt.tags:
                for frame_idx, frame_info in tag.additional_info.get("frame_tags", {}).items():
                    frame_idx = int(frame_idx)
                    if frame_idx not in frame_tags:
                        frame_tags[frame_idx] = {}
                    
                    if feature not in frame_tags[frame_idx]:
                        frame_tags[frame_idx][feature] = []

                    assert "box" in frame_info
                    
                    frame_tags[frame_idx][feature].append(FrameTag(
                        text=tag.text,
                        box=frame_info["box"],
                        confidence=frame_info.get("confidence")
                    ))

        return frame_tags

    def _feature_to_track(self, feature: str) -> str:
        return self._label_to_track(self._feature_to_label(feature))

    def _feature_to_label(self, feature: str) -> str:
        if feature in self.cfg.name_mapping:
            return self.cfg.name_mapping[feature]
        return feature.replace("_", " ").title()

    def _label_to_track(self, label: str) -> str:
        return label.lower().replace(" ", "_")

    def _aggregate_video_tags(self, track_to_tags: dict[str, list[VideoTag]], intervals: list[tuple[int, int]]) -> list[AggTag]:
        all_tags = deepcopy(track_to_tags)
        for track, tags in track_to_tags.items():
            all_tags[track] = sorted(tags, key=lambda x: x.start_time)

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

        for agg_tag in result:
            for feat in self.cfg.coalesce_tracks:
                agg_tag.coalesce(feat)

        for agg_tag in result:
            for feat in self.cfg.single_tag_tracks:
                agg_tag.keep_longest(feat)
        
        return result
    
    def _get_shot_intervals(self, shot_tags: list[Tag]) -> list[tuple[int, int]]:
        if not shot_tags:
            return []
        intvs = []
        shot_tags = sorted(shot_tags, key=lambda x: x.start_time)
        last_source = shot_tags[0].source
        for tag in shot_tags:
            if tag.source == last_source:
                intvs.append((tag.start_time, tag.end_time))
            else:
                # merge intervals from previous source
                intvs[-1] = (intvs[-1][0], tag.end_time)
                last_source = tag.source
        return intvs

    def _get_sentence_intervals(self, tags: list[VideoTag]) -> list[tuple[int, int]]:
        tags = sorted(tags, key=lambda x: x.start_time)
        sentence_delimiters = ['.', '?', '!']
        intervals = []
        if len(tags) == 0:
            return []
        quiet = True
        curr_int = [0]
        fake_sentence_cutoff = self.cfg.max_sentence_words
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
                fake_sentence_cutoff = i + self.cfg.max_sentence_words
                # end and commit the speaking interval, add one due to exclusive bounds
                curr_int.append(tag.end_time+1)
                intervals.append((curr_int[0], curr_int[-1]))
                curr_int.clear()
                # start a new silent interval
                curr_int.append(tag.end_time+1)
                quiet = True
        return intervals

    def dump_tracks(
        self,
        tc: TrackCollection
    ) -> dict:
        result = {"version": 1, "metadata_tags": {}}
        
        # add aggregated tags
        for key, tags in tc.agg_tracks.items():
            label = self._feature_to_label(key)
            if key not in result["metadata_tags"]:
                result["metadata_tags"][key] = {"label": label, "tags": []}

            for agg_tag in tags:
                entry = {
                    "start_time": agg_tag.start_time,
                    "end_time": agg_tag.end_time,
                    "text": defaultdict(list)
                }
                
                for track, video_tags in agg_tag.tags.items():
                    track_label = self._feature_to_label(track)
                        
                    for vtag in video_tags:
                        as_dict = asdict(vtag)
                        if vtag.text is not None:
                            # NOTE: this is just a tag file convention, probably should just be a string value
                            as_dict["text"] = [as_dict["text"]]
                        assert isinstance(entry["text"], defaultdict)
                        entry["text"][track_label].append(as_dict)
                        
                result["metadata_tags"][key]["tags"].append(entry)

        # add standalone tracks
        for key, video_tags in tc.tracks.items():
            label = self._feature_to_label(key)
            track = self._label_to_track(label)

            if track not in result["metadata_tags"]:
                result["metadata_tags"][track] = {"label": label, "tags": []}

            for vtag in video_tags:
                entry: dict[str, object] = {
                    "start_time": vtag.start_time,
                    "end_time": vtag.end_time,
                }
                if vtag.text is not None:
                    entry["text"] = vtag.text
                result["metadata_tags"][track]["tags"].append(entry)

        return result

    def dump_overlay(self, overlay: Overlay) -> dict:
    
        result = {}
        for frame_idx, feature_map in overlay.items():
            frame_idx = str(frame_idx)
            if frame_idx not in result:
                result[frame_idx] = {}
            for feature, ftags in feature_map.items():
                label = self._feature_to_track(feature)
                if label not in result[frame_idx]:
                    result[frame_idx][label] = {"tags": []}
                for ftag in ftags:
                    as_dict = asdict(ftag)
                    result[frame_idx][label]["tags"].append(as_dict)

        return {"version": 1, "overlay_tags": {"frame_level_tags": result}}