from dataclasses import asdict
from datetime import datetime
import shutil
import json
import os
import base64
import time

from src.common.content import Content
from src.tags.tagstore.types import *
from src.tags.tagstore.abstract import Tagstore

class FilesystemTagStore(Tagstore):
    def __init__(self, base_dir: str):
        self.base_path = base_dir
        os.makedirs(self.base_path, exist_ok=True)

    def start_job(self,
        qhit: str,
        track: str,
        stream: str,
        author: str,
        q: Content | None = None
    ) -> UploadJob:
        """
        Starts a new job with provided metadata
        """
        jobid = qhit + "/" + track + "/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = self._get_job_dir(jobid)
        os.makedirs(job_dir, exist_ok=True)

        job = UploadJob(
            id=jobid,
            qhit=qhit,
            track=track,
            stream=stream,
            timestamp=time.time(),
            author=author
        )

        metadata_path = self._get_job_metadata_path(jobid)
        with open(metadata_path, 'w') as f:
            json.dump(asdict(job), f, indent=2)

        return job

    def upload_tags(self, 
        tags: list[Tag], 
        jobid: str,
        q: Content | None = None
    ) -> None:
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

    def find_tags(self, q: Content | None = None, **filters) -> list[Tag]:
        """
        Find tags with flexible filtering.
        
        Supported filters:
        - stream: str  
        - track: str
        - jobid: str  
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
        
        if 'jobid' in filters:
            # If specific jobid requested, only check that job
            job_ids = [filters['jobid']]
        else:
            # Get all matching jobs
            job_ids = self.find_jobs(**job_filters)
        
        # Collect tags from matching jobs
        for job_id in job_ids:
            tags = self._get_tags_for_job(job_id)
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

    def find_jobs(self, q: Content | None = None, **filters) -> list[str]:
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
        
        for job_id, job_dir in self._get_job_ids_with_paths():
            
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

    def delete_job(self, jobid: str, q: Content | None = None) -> None:
        dir = self._get_job_dir(jobid)
        shutil.rmtree(dir, ignore_errors=True)

    def count_tags(self, q: Content | None = None, **filters) -> int:
        """Count tags matching the given filters without loading all data"""
        return len(self.find_tags(**filters))

    def count_jobs(self, q: Content | None = None, **filters) -> int:
        """Count jobs matching the given filters"""
        return len(self.find_jobs(**filters))

    def get_job(self, jobid: str, q: Content | None=None) -> UploadJob | None:
        """
        Get job metadata
        """
        metadata_path = self._get_job_metadata_path(jobid)
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            job_data = json.load(f)
            return UploadJob(**job_data)

    def _get_job_ids_with_paths(self) -> list[tuple[str, str]]:
        """Get all job IDs with their corresponding paths"""
        if not os.path.exists(self.base_path):
            return []
        
        # jobids are represented by qhit/feature/stream_name, return this path
        job_ids = []
        for qhit in os.listdir(self.base_path):
            for feature in os.listdir(os.path.join(self.base_path, qhit)):
                for stream_name in os.listdir(os.path.join(self.base_path, qhit, feature)):
                    job_ids.append(f"{qhit}/{feature}/{stream_name}")

        job_dirs = [os.path.join(self.base_path, job_id) for job_id in job_ids]
        return list(zip(job_ids, job_dirs))

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

    def _get_tags_for_job(self, job_id: str) -> list[Tag]:
        """Helper method to get all tags for a specific job without filtering"""
        all_tags = []
        job_dir = self._get_job_dir(job_id)
        
        if not os.path.exists(job_dir):
            return all_tags
        
        # Iterate through all tag files in the job directory
        for filename in os.listdir(job_dir):
            if filename.endswith('.json') and filename != 'jobmetadata.json':
                tags_path = os.path.join(job_dir, filename)
                try:
                    with open(tags_path, 'r') as f:
                        tag_data = json.load(f)
                        tags = [Tag(**tag_dict) for tag_dict in tag_data]
                        all_tags.extend(tags)
                except Exception:
                    # Skip corrupted files
                    continue
        
        return all_tags