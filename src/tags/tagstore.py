from dataclasses import asdict
import json
import os
import base64
import tempfile
from typing import List, Dict, Tuple
from collections import defaultdict

from common_ml.tags import VideoTag, FrameTag
from common_ml.utils.metrics import timeit
from elv_client_py import ElvClient
from loguru import logger

from src.common.schema import Tag, UploadJob
from src.common.content import Content
from src.tags.legacy.agg import (
    aggregate_video_tags, 
    format_tracks, 
    _get_sentence_intervals
)

class FilesystemTagStore:
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def start_job(self, job: UploadJob) -> None:
        """
        Starts a new job with provided metadata
        """
        job_dir = self._get_job_dir(job.id)
        os.makedirs(job_dir, exist_ok=True)
        
        metadata_path = self._get_job_metadata_path(job.id)
        with open(metadata_path, 'w') as f:
            json.dump(asdict(job), f, indent=2)

    def upload_tags(self, tags: list[Tag], jobid: str) -> None:
        """
        Upload tags for a specific job, grouped by source
        """
        if not tags:
            return
        
        job_dir = self._get_job_dir(jobid)
        if not os.path.exists(job_dir):
            raise ValueError(f"Job {jobid} not found. Call start_job() first.")
        
        # Group tags by source
        tags_by_source = {}
        for tag in tags:
            if tag.source not in tags_by_source:
                tags_by_source[tag.source] = []
            tags_by_source[tag.source].append(tag)
        
        # Write each source group to its own file
        for source, source_tags in tags_by_source.items():
            tags_path = self._get_tags_path(jobid, source)
            
            # Load existing tags if file exists
            existing_tags = []
            if os.path.exists(tags_path):
                try:
                    with open(tags_path, 'r') as f:
                        existing_data = json.load(f)
                        existing_tags = [Tag(**tag_data) for tag_data in existing_data]
                except Exception:
                    # If file is corrupted, start fresh
                    existing_tags = []
            
            # Combine existing and new tags
            all_tags = existing_tags + source_tags
            
            # Write back to file atomically
            temp_path = tags_path + ".tmp"
            try:
                with open(temp_path, 'w') as f:
                    json.dump([asdict(tag) for tag in all_tags], f, indent=2)
                os.rename(temp_path, tags_path)
            except Exception as e:
                # Clean up temp file if write failed
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e

    def find_tags(self, **filters) -> List[Tag]:
        """
        Find tags with flexible filtering.
        
        Supported filters:
        - qhit: str
        - stream: str  
        - track: str
        - job_id: str
        - sources: List[str] (tags with source in this list)
        - start_time_gte: float
        - start_time_lte: float
        - text_contains: str
        - author: str
        - limit: int
        - offset: int
        """
        all_tags = []
        
        # First, get all jobs that match job-level filters
        job_filters = {}
        if 'qhit' in filters:
            job_filters['qhit'] = filters['qhit']
        if 'stream' in filters:
            job_filters['stream'] = filters['stream']
        if 'track' in filters:
            job_filters['track'] = filters['track']
        if 'author' in filters:
            job_filters['author'] = filters['author']
        
        if 'job_id' in filters:
            # If specific job_id requested, only check that job
            job_ids = [filters['job_id']]
        else:
            # Get all matching jobs
            job_ids = self.find_jobs(**job_filters)
        
        # Collect tags from matching jobs
        for job_id in job_ids:
            tags = self.get_tags(job_id)
            all_tags.extend(tags)
        
        # Apply tag-level filters
        filtered_tags = []
        for tag in all_tags:
            # Source filter
            if 'sources' in filters:
                if tag.source not in filters['sources']:
                    continue
            
            # Time range filters
            if 'start_time_gte' in filters:
                if tag.start_time < filters['start_time_gte']:
                    continue
            
            if 'start_time_lte' in filters:
                if tag.start_time > filters['start_time_lte']:
                    continue
            
            # Text search
            if 'text_contains' in filters:
                if filters['text_contains'].lower() not in tag.text.lower():
                    continue
            
            filtered_tags.append(tag)
        
        # Apply pagination
        if 'offset' in filters:
            offset = filters['offset']
            filtered_tags = filtered_tags[offset:]
        
        if 'limit' in filters:
            limit = filters['limit']
            filtered_tags = filtered_tags[:limit]
        
        return filtered_tags

    def find_jobs(self, **filters) -> List[str]:
        """
        Find job IDs with flexible filtering.
        
        Supported filters:
        - qhit: str
        - stream: str
        - track: str 
        - author: str
        - timestamp_gte: float
        - timestamp_lte: float
        - limit: int
        - offset: int
        """
        job_ids = []
        
        # Iterate through all directories in base_path
        if not os.path.exists(self.base_path):
            return job_ids
        
        for job_id in os.listdir(self.base_path):
            job_dir = os.path.join(self.base_path, job_id)
            
            # Skip if not a directory
            if not os.path.isdir(job_dir):
                continue
            
            # Get job metadata to check filters
            job = self.get_job(job_id)
            if job is None:
                continue
            
            # Apply filters
            if 'qhit' in filters and job.qhit != filters['qhit']:
                continue
            if 'track' in filters and job.track != filters['track']:
                continue
            if 'stream' in filters and job.stream != filters['stream']:
                continue
            if 'author' in filters and job.author != filters['author']:
                continue
            if 'timestamp_gte' in filters and job.timestamp < filters['timestamp_gte']:
                continue
            if 'timestamp_lte' in filters and job.timestamp > filters['timestamp_lte']:
                continue
            
            job_ids.append(job_id)
        
        # Apply pagination
        if 'offset' in filters:
            offset = filters['offset']
            job_ids = job_ids[offset:]
        
        if 'limit' in filters:
            limit = filters['limit']
            job_ids = job_ids[:limit]
        
        return job_ids

    def count_tags(self, **filters) -> int:
        """Count tags matching the given filters without loading all data"""
        return len(self.find_tags(**filters))

    def count_jobs(self, **filters) -> int:
        """Count jobs matching the given filters"""
        return len(self.find_jobs(**filters))

    def get_tags_for_job(self, job_id: str) -> List[Tag]:
        """Get all tags for a specific job"""
        return self.find_tags(job_id=job_id)

    def get_job(self, job_id: str) -> UploadJob | None:
        """
        Get job metadata
        """
        metadata_path = self._get_job_metadata_path(job_id)
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            job_data = json.load(f)
            return UploadJob(**job_data)

    def get_tags(self, job_id: str) -> list[Tag]:
        """
        Get all tags for a specific job (from all source files)
        """
        job_dir = self._get_job_dir(job_id)
        
        if not os.path.exists(job_dir):
            return []
        
        all_tags = []
        
        # Find all .json files that aren't jobmetadata.json
        for filename in os.listdir(job_dir):
            if filename.endswith('.json') and filename != 'jobmetadata.json':
                tags_path = os.path.join(job_dir, filename)
                
                with open(tags_path, 'r') as f:
                    tags_data = json.load(f)
                    tags = [Tag(**tag_data) for tag_data in tags_data]
                    all_tags.extend(tags)

        return all_tags

    def commit(self, q: Content, interval: int = 10) -> None:
        """
        Format and upload tags for a content object using tags stored in the tagstore.
        
        Args:
            q: Content object with write token
            interval: Interval in minutes to bucket the formatted results (default: 10)
        """
        logger.info(f"Starting commit for content {q.qhit}")
        
        # Get tags for this content
        tags = self._get_latest_tags_for_content(q.qhit)
        if not tags:
            logger.info(f"No tags found for content {q.qhit}")
            return

        # Parse tags into video and frame formats
        all_video_tags, all_frame_tags = self._parse_tags(tags)
        
        if not all_video_tags:
            logger.info(f"No tags found for content {q.qhit}")
            return
        
        # Process tags
        formatted_tracks, overlays = self._format_tags_for_upload(
            all_video_tags, all_frame_tags, interval
        )
        
        # Upload formatted tags
        self._upload_formatted_tags(q, formatted_tracks, overlays)
        
        logger.info(f"Successfully committed tags for content {q.qhit}")

    def _get_latest_tags_for_content(self, qhit: str) -> List[Tag]:
        """Get tags from the latest job for each source for the given content."""
        job_ids = self.find_jobs(qhit=qhit)
        if not job_ids:
            return []

        jobs = [self.get_job(job_id) for job_id in job_ids]
        jobs = [job for job in jobs if job is not None]
        jobs.sort(key=lambda job: job.id, reverse=True)  # Newest first

        tags = []
        sources_tagged = set()

        # For each job (newest first), collect tags for sources we haven't seen
        for job in jobs:
            job_sources = set()
            new_tags = []
            
            for tag in self.get_tags(job.id):
                if tag.source in sources_tagged:
                    continue
                job_sources.add(tag.source)
                new_tags.append(tag)
            
            sources_tagged.update(job_sources)
            tags.extend(new_tags)

        return tags

    def _format_tags_for_upload(
        self, 
        all_video_tags: Dict[str, List[VideoTag]], 
        all_frame_tags: Dict[str, Dict[int, List[FrameTag]]], 
        interval: int
    ) -> Tuple[List[dict], List[dict]]:
        """Format tags into tracks and overlays for upload."""
        
        # Handle shot intervals for aggregation
        shot_intervals = []
        if "shot" in all_video_tags:
            shot_intervals = [(tag.start_time, tag.end_time) for tag in all_video_tags["shot"]]

        # Aggregate tags by shot intervals if available
        aggshot_tags = {}
        if shot_intervals:
            non_shot_tags = {f: tags for f, tags in all_video_tags.items() if f != "shot"}
            aggshot_tags = {
                "shot_tags": aggregate_video_tags(non_shot_tags, shot_intervals)
            }

        # Handle automatic captions from ASR
        if "asr" in all_video_tags:
            try:
                sentence_intervals = _get_sentence_intervals(all_video_tags["asr"])
                sentence_agg_tags = aggregate_video_tags(
                    {"asr": all_video_tags["asr"]}, 
                    sentence_intervals
                )
                stt_sent_track = [
                    VideoTag(agg_tag.start_time, agg_tag.end_time, agg_tag.tags["asr"][0].text) 
                    for agg_tag in sentence_agg_tags if "asr" in agg_tag.tags
                ]
                all_video_tags["auto_captions"] = stt_sent_track
            except Exception as e:
                logger.warning(f"Failed to create auto captions from ASR: {e}")

        # Format tracks and overlays
        try:
            formatted_tracks = format_tracks(
                aggshot_tags, 
                all_video_tags, 
                interval, 
                custom_labels={}
            )
            # Remove fps parameter since frame timestamps are already global
            overlays = format_overlay(all_frame_tags, interval)
            return formatted_tracks, overlays
        except Exception as e:
            logger.error(f"Failed to format tracks/overlays: {e}")
            raise

    def _upload_formatted_tags(
        self, 
        q: Content, 
        formatted_tracks: List[dict], 
        overlays: List[dict]
    ) -> None:
        """Upload formatted tracks and overlays to fabric."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            to_upload = []
            
            # Write formatted tracks
            for i, track in enumerate(formatted_tracks):
                fpath = os.path.join(temp_dir, f"video-tags-tracks-{i:04d}.json")
                to_upload.append(fpath)
                with open(fpath, 'w') as f:
                    json.dump(track, f)

            # Write overlays
            for i, overlay in enumerate(overlays):
                fpath = os.path.join(temp_dir, f"video-tags-overlay-{i:04d}.json")
                to_upload.append(fpath)
                with open(fpath, 'w') as f:
                    json.dump(overlay, f)

            if not to_upload:
                logger.info("No formatted tags to upload")
                return

            # Create upload jobs
            jobs = [
                ElvClient.FileJob(
                    local_path=path, 
                    out_path=f"video_tags/{os.path.basename(path)}", 
                    mime_type="application/json"
                ) 
                for path in to_upload
            ]

            # Upload files
            try:
                with timeit("Uploading aggregated files"):
                    q.upload_files(file_jobs=jobs, finalize=False)
                logger.info(f"Uploaded {len(jobs)} formatted tag files")
            except Exception as e:
                logger.error(f"Failed to upload files: {e}")
                raise

            # Add metadata links
            try:
                with timeit("Adding links"):
                    self._add_links(q, [os.path.basename(path) for path in to_upload])
                logger.info("Successfully added metadata links")
            except Exception as e:
                logger.error(f"Failed to add metadata links: {e}")
                raise

    def _parse_tags(self, tags: List[Tag]) -> Tuple[Dict[str, List[VideoTag]], Dict[str, Dict[int, List[FrameTag]]]]:
        """Convert tagstore tags to VideoTag and FrameTag formats."""
        vtags, ftags = {}, {}
        
        for tag in tags:
            job = self.get_job(tag.jobid)
            if not job:
                logger.warning(f"Could not find job {tag.jobid} for tag")
                continue
                
            feature = job.track
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
        
        return vtags, ftags

    def _add_links(self, q: Content, filenames: List[str]) -> None:
        """Add metadata links for uploaded tag files."""
        data = {}
        
        for filename in filenames:
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
        
        if data:
            q.merge_metadata(metadata=data, metadata_subtree='video_tags')

    def _encode_source_for_filename(self, source: str) -> str:
        """Encode source name for safe filesystem usage"""
        if '/' in source:
            # Base64 encode sources with slashes
            encoded = base64.b64encode(source.encode('utf-8')).decode('ascii')
            return f"b64_{encoded}"
        return source

    def _decode_source_from_filename(self, filename: str) -> str:
        """Decode source name from filesystem filename"""
        if filename.startswith('b64_'):
            # Remove b64_ prefix and decode
            encoded = filename[4:]  # Remove 'b64_' prefix
            return base64.b64decode(encoded.encode('ascii')).decode('utf-8')
        return filename

    def _get_job_dir(self, job_id: str) -> str:
        """Get the directory path for a specific job"""
        return os.path.join(self.base_path, job_id)

    def _get_job_metadata_path(self, job_id: str) -> str:
        """Get the path to job metadata file"""
        return os.path.join(self._get_job_dir(job_id), "jobmetadata.json")

    def _get_tags_path(self, job_id: str, source: str) -> str:
        """Get the path to tags file for a specific source"""
        encoded_source = self._encode_source_for_filename(source)
        return os.path.join(self._get_job_dir(job_id), f"{encoded_source}.json")

def format_overlay(all_frame_tags: Dict[str, Dict[int, List[FrameTag]]], interval: int) -> List[Dict[str, Dict[int, List[FrameTag]]]]:
    if len(all_frame_tags) == 0:
        return []
    buckets = defaultdict(lambda: {"version": 1, "overlay_tags": {"frame_level_tags": defaultdict(dict)}})
    interval = interval*60*1000  # Convert to milliseconds
    
    for feature, frame_tags in all_frame_tags.items():
        label = feature_to_label(feature)
        for frame_idx, ftags in frame_tags.items():
            # Assume frame_idx is already global timestamp in milliseconds
            timestamp_ms = frame_idx
            bucket_idx = int(timestamp_ms/interval)
            buckets[bucket_idx]["overlay_tags"]["frame_level_tags"][frame_idx][label_to_track(label)] = {"tags": [asdict(tag) for tag in ftags]}
    
    # Add timestamps (frame_idx is already in milliseconds)
    for bucket in buckets.values():
        for frame_idx in bucket["overlay_tags"]["frame_level_tags"]:
            bucket["overlay_tags"]["frame_level_tags"][frame_idx]["timestamp_sec"] = frame_idx
    
    buckets = [buckets[i] if i in buckets else {"version": 1, "overlay_tags": {"frame_level_tags": {}}} for i in range(max(buckets.keys())+1)]
    return buckets


# TODO: we should really centralize this somewhere
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