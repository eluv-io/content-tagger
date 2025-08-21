from dataclasses import dataclass, asdict
import json
import os

@dataclass
class Tag:
    start_time: int
    end_time: int
    text: str
    additional_info: dict
    source: str
    jobid: str

@dataclass
class Job:
    id: str
    qhit: str
    stream: str | None
    track: str
    timestamp: float
    author: str

class FilesystemTagStore:
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _get_job_dir(self, job_id: str) -> str:
        """Get the directory path for a specific job"""
        return os.path.join(self.base_path, job_id)

    def _get_job_metadata_path(self, job_id: str) -> str:
        """Get the path to job metadata file"""
        return os.path.join(self._get_job_dir(job_id), "jobmetadata.json")

    def _get_tags_path(self, job_id: str, source: str) -> str:
        """Get the path to tags file for a specific source"""
        return os.path.join(self._get_job_dir(job_id), f"{source}.json")

    def start_job(self, job: Job) -> None:
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

    def get_job(self, job_id: str) -> Job | None:
        """
        Get job metadata
        """
        metadata_path = self._get_job_metadata_path(job_id)
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            job_data = json.load(f)
            return Job(**job_data)

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

    def get_jobs(
        self,
        qhit: str | None = None,
        track: str | None = None, 
        stream: str | None = None,
        auth: str | None = None
    ) -> list[str]:
        """
        Get jobids based on the filters
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
            if qhit is not None and job.qhit != qhit:
                continue
            if track is not None and job.track != track :
                continue
            if stream is not None and job.stream != stream:
                continue
            if auth is not None and job.author != auth:
                continue
            
            job_ids.append(job_id)
        
        return job_ids

    # TODO: better way of querying this than 3 args
    def list_tagged_sources(self, qhit: str, track: str, stream: str) -> list[str]:
        """
        List all sources where author is "tagger" from any job
        """

        tagged_sources = set()

        jobids = self.get_jobs(qhit=qhit, auth="tagger", track=track, stream=stream)

        for job_id in jobids:
            tagged_sources |= {tag.source for tag in self.get_tags(job_id)}

        return list(tagged_sources)