import time
import requests
from datetime import datetime
from src.tags.tagstore.types import Tag, UploadJob
from src.tags.tagstore.abstract import Tagstore

class RestTagstore(Tagstore):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

    def _get_headers(self, auth: str | None) -> dict:
        """Get headers with auth if provided"""
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = f'Bearer {auth}'
        return headers

    def start_job(self,
        qhit: str,
        track: str,
        stream: str,
        author: str,
        auth: str | None = None
    ) -> UploadJob:
        """
        Starts a new job with provided metadata
        """
        # Create job via REST API
        job_data = {
            "track": track,
            "author": author,
            "additional_info": {
                "stream": stream,
                "qhit": qhit
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/{qhit}/jobs", 
            json=job_data,
            headers=self._get_headers(auth)
        )
        response.raise_for_status()
        
        result = response.json()
        job_id = result["job_id"]
        
        # Create UploadJob object with the returned job_id
        job = UploadJob(
            id=job_id,
            qhit=qhit,
            track=track,
            stream=stream,
            timestamp=time.time(),
            author=author
        )
        
        return job

    def upload_tags(self, tags: list[Tag], jobid: str, auth: str | None = None) -> None:
        """
        Upload tags for a specific job
        """
        if not tags:
            return
        
        # Extract qhit from jobid (assuming format: qhit/track/timestamp)
        qhit = jobid.split('/')[0]
        
        # Convert tags to API format
        api_tags = []
        for tag in tags:
            api_tag = {
                "start_time": tag.start_time,
                "end_time": tag.end_time,
                "tag": tag.text,
                "source": tag.source,
                "additional_properties": tag.additional_info
            }
            api_tags.append(api_tag)
        
        # Upload tags
        upload_data = {
            "job_id": jobid,
            "tags": api_tags
        }
        
        response = self.session.post(
            f"{self.base_url}/{qhit}/tags", 
            json=upload_data,
            headers=self._get_headers(auth)
        )
        response.raise_for_status()

    def find_tags(self, auth: str | None = None, **filters) -> list[Tag]:
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
        if 'qhit' not in filters:
            raise ValueError("qhit is required for REST API queries")
        
        qhit = filters['qhit']
        
        # Build query parameters
        params = {}
        
        if 'job_id' in filters:
            params['job_id'] = filters['job_id']
        if 'track' in filters:
            params['track'] = filters['track']
        if 'author' in filters:
            params['author'] = filters['author']
        if 'start_time_gte' in filters:
            params['start_time_gte'] = int(filters['start_time_gte'])
        if 'start_time_lte' in filters:
            params['start_time_lte'] = int(filters['start_time_lte'])
        if 'text_contains' in filters:
            params['text_contains'] = filters['text_contains']
        if 'limit' in filters:
            params['limit'] = filters['limit']
        if 'offset' in filters:
            params['start'] = filters['offset']
        
        response = self.session.get(
            f"{self.base_url}/{qhit}/tags", 
            params=params,
            headers=self._get_headers(auth)
        )
        response.raise_for_status()
        
        result = response.json()
        api_tags = result.get('tags', [])
        
        # Convert API tags to Tag objects
        tags = []
        for api_tag in api_tags:
            tag = Tag(
                start_time=api_tag['start_time'],
                end_time=api_tag['end_time'],
                text=api_tag['tag'],
                additional_info=api_tag.get('additional_properties', {}),
                source=api_tag.get('source', 'content'),
                jobid=api_tag['job_id']
            )
            
            # Apply source filter if specified
            if 'sources' in filters:
                if tag.source not in filters['sources']:
                    continue
            
            tags.append(tag)
        
        return tags

    def find_jobs(self, auth: str | None = None, **filters) -> list[str]:
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
        if 'qhit' not in filters:
            raise ValueError("qhit is required for REST API queries")
        
        qhit = filters['qhit']
        
        # Build query parameters
        params = {}
        
        if 'track' in filters:
            params['track'] = filters['track']
        if 'author' in filters:
            params['author'] = filters['author']
        if 'limit' in filters:
            params['limit'] = filters['limit']
        if 'offset' in filters:
            params['start'] = filters['offset']
        
        response = self.session.get(
            f"{self.base_url}/{qhit}/jobs", 
            params=params,
            headers=self._get_headers(auth)
        )
        response.raise_for_status()
        
        result = response.json()
        jobs = result.get('jobs', [])
        
        # Extract job IDs
        job_ids = [str(job['id']) for job in jobs]
        
        return job_ids

    def count_tags(self, auth: str | None = None, **filters) -> int:
        """Count tags matching the given filters"""
        # Use the same query but with a high limit to get count from meta
        query_filters = filters.copy()
        query_filters['limit'] = 1  # Just need meta info
        
        if 'qhit' not in query_filters:
            raise ValueError("qhit is required for REST API queries")
        
        qhit = query_filters['qhit']
        
        # Build query parameters
        params = {}
        
        if 'job_id' in query_filters:
            params['job_id'] = query_filters['job_id']
        if 'track' in query_filters:
            params['track'] = query_filters['track']
        if 'author' in query_filters:
            params['author'] = query_filters['author']
        if 'start_time_gte' in query_filters:
            params['start_time_gte'] = int(query_filters['start_time_gte'])
        if 'start_time_lte' in query_filters:
            params['start_time_lte'] = int(query_filters['start_time_lte'])
        if 'text_contains' in query_filters:
            params['text_contains'] = query_filters['text_contains']
        params['limit'] = 1
        
        response = self.session.get(
            f"{self.base_url}/{qhit}/tags", 
            params=params,
            headers=self._get_headers(auth)
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('meta', {}).get('total', 0)

    def count_jobs(self, auth: str | None = None, **filters) -> int:
        """Count jobs matching the given filters"""
        # Use the same query but with a high limit to get count from meta
        query_filters = filters.copy()
        query_filters['limit'] = 1  # Just need meta info
        
        if 'qhit' not in query_filters:
            raise ValueError("qhit is required for REST API queries")
        
        qhit = query_filters['qhit']
        
        # Build query parameters
        params = {}
        
        if 'track' in query_filters:
            params['track'] = query_filters['track']
        if 'author' in query_filters:
            params['author'] = query_filters['author']
        params['limit'] = 1
        
        response = self.session.get(
            f"{self.base_url}/{qhit}/jobs", 
            params=params,
            headers=self._get_headers(auth)
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('meta', {}).get('total', 0)

    def get_job(self, jobid: str, auth: str | None = None) -> UploadJob | None:
        """
        Get job metadata
        """
        # Extract qhit from jobid (assuming format: qhit/track/timestamp)
        qhit = jobid.split('/')[0]
        
        try:
            response = self.session.get(
                f"{self.base_url}/{qhit}/jobs/{jobid}",
                headers=self._get_headers(auth)
            )
            response.raise_for_status()
            
            job_data = response.json()
            
            # Convert API job to UploadJob
            job = UploadJob(
                id=str(job_data['id']),
                qhit=qhit,
                track=job_data['track'],
                stream=job_data.get('additional_info', {}).get('stream'),
                timestamp=datetime.fromisoformat(job_data['timestamp']).timestamp(),
                author=job_data['author']
            )
            
            return job
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise