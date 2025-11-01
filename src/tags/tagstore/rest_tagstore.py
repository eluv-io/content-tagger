import time
import json
import requests
from dateutil import parser
from src.common.logging import logger

from src.common.content import Content
from src.tags.tagstore.model import Tag, Batch
from src.tags.tagstore.abstract import Tagstore

class RestTagstore(Tagstore):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

    def _get_headers(self, q: Content | None) -> dict:
        """Get headers with q if provided"""
        headers = {'Content-Type': 'application/json'}
        assert q is not None
        headers['Authorization'] = f"Bearer {q.token()}"
        return headers

    def _log_response_and_raise(self, response: requests.Response):
        """Log response content before raising HTTPError"""
        try:
            response_json = response.json()
            logger.error(f"{json.dumps(response_json)}")
        except Exception:
            logger.error(f"HTTP {response.status_code} response (non-JSON): {response.text}")
        response.raise_for_status()

    def create_batch(self,
        qhit: str,
        track: str,
        stream: str,
        author: str,
        q: Content | None = None
    ) -> Batch:
        """
        Starts a new batch with provided metadata
        """
        # Create batch via REST API
        batch_data = {
            "track": track,
            "author": author,
            "additional_info": {
                "stream": stream,
                "qhit": qhit
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/{qhit}/batches", 
            json=batch_data,
            headers=self._get_headers(q)
        )
        
        if not response.ok:
            self._log_response_and_raise(response)
        
        result = response.json()
        batch_id = result["batch_id"]
        
        # Create Batch object with the returned batch_id
        batch = Batch(
            id=batch_id,
            qhit=qhit,
            track=track,
            stream=stream,
            timestamp=time.time(),
            author=author
        )
        
        return batch

    def upload_tags(self, tags: list[Tag], batch_id: str, q: Content | None = None) -> None:
        """
        Upload tags for a specific batch
        """
        if not tags:
            return
        
        assert q is not None
        qhit = q.qid
        
        # Convert tags to API format
        api_tags = []
        for tag in tags:
            api_tag = {
                "start_time": tag.start_time,
                "end_time": tag.end_time,
                "tag": tag.text,
                "source": tag.source,
                "additional_info": tag.additional_info
            }
            api_tags.append(api_tag)
        
        # Upload tags
        upload_data = {
            "batch_id": batch_id,
            "tags": api_tags
        }
        
        response = self.session.post(
            f"{self.base_url}/{qhit}/tags", 
            json=upload_data,
            headers=self._get_headers(q)
        )
        
        if not response.ok:
            self._log_response_and_raise(response)

    def find_tags(self, q: Content | None = None, **filters) -> list[Tag]:
        """
        Find tags with flexible filtering.
        
        Supported filters:
        - qhit: str
        - stream: str  
        - track: str
        - batch_id: str
        - sources: List[str] (tags with source in this list)
        - start_time_gte: float
        - start_time_lte: float
        - text_contains: str
        - author: str
        - limit: int
        - offset: int
        """
        
        assert q is not None
        if 'qhit' in filters:
            assert filters['qhit'] == q.qid
        qhit = q.qid
        
        # Build query parameters
        params = {}
        
        if 'batch_id' in filters:
            # TODO: probably should change to batch_id everywhere
            params['batch_id'] = filters['batch_id']
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

        # if no limit, then specify a high limit to avoid default limit
        if 'limit' not in params:
            params['limit'] = 100000
        
        response = self.session.get(
            f"{self.base_url}/{qhit}/tags", 
            params=params,
            headers=self._get_headers(q)
        )
        
        if not response.ok:
            self._log_response_and_raise(response)
        
        result = response.json()
        api_tags = result.get('tags', [])
        
        # Convert API tags to Tag objects
        tags = []
        for api_tag in api_tags:
            tag = Tag(
                start_time=api_tag['start_time'],
                end_time=api_tag['end_time'],
                text=api_tag['tag'],
                additional_info=api_tag.get('additional_info', {}),
                source=api_tag.get('source', ''),
                batch_id=api_tag['batch_id']
            )
            
            # Apply source filter if specified
            if 'sources' in filters:
                if tag.source not in filters['sources']:
                    continue
            
            tags.append(tag)
        
        return tags

    def find_batches(self, q: Content | None = None, **filters) -> list[str]:
        """
        Find batch IDs with flexible filtering.
        
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
        
        assert q is not None
        if 'qhit' in filters:
            assert filters['qhit'] == q.qid
        qhit = q.qid
        
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
            f"{self.base_url}/{qhit}/batches", 
            params=params,
            headers=self._get_headers(q)
        )
        
        if not response.ok:
            self._log_response_and_raise(response)
        
        result = response.json()
        batches = result.get('batches', [])
        
        # Extract batch IDs
        batch_ids = [str(batch['id']) for batch in batches]
        
        return batch_ids

    def count_tags(self, q: Content | None = None, **filters) -> int:
        """Count tags matching the given filters"""
        # Use the same query but with a high limit to get count from meta
        query_filters = filters.copy()
        query_filters['limit'] = 1  # Just need meta info

        assert q is not None        
        qhit = q.qid
        
        # Build query parameters
        params = {}
        
        if 'batch_id' in query_filters:
            params['batch_id'] = query_filters['batch_id']
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
            headers=self._get_headers(q)
        )
        
        if not response.ok:
            self._log_response_and_raise(response)
        
        result = response.json()
        return result.get('meta', {}).get('total', 0)

    def count_batches(self, q: Content | None = None, **filters) -> int:
        """Count batches matching the given filters"""
        # Use the same query but with a high limit to get count from meta
        query_filters = filters.copy()
        query_filters['limit'] = 1  # Just need meta info
        
        assert q is not None
        
        qhit = q.qid
        
        # Build query parameters
        params = {}
        
        if 'track' in query_filters:
            params['track'] = query_filters['track']
        if 'author' in query_filters:
            params['author'] = query_filters['author']
        params['limit'] = 1
        
        response = self.session.get(
            f"{self.base_url}/{qhit}/batches", 
            params=params,
            headers=self._get_headers(q)
        )
        
        if not response.ok:
            self._log_response_and_raise(response)
        
        result = response.json()
        return result.get('meta', {}).get('total', 0)

    def get_batch(self, batch_id: str, q: Content | None = None) -> Batch | None:
        """
        Get batch metadata
        """
        # Extract qhit from batch_id (assuming format: qhit/track/timestamp)
        assert q is not None
        qhit = q.qid
        
        try:
            response = self.session.get(
                f"{self.base_url}/{qhit}/batches/{batch_id}",
                headers=self._get_headers(q)
            )
            
            if response.status_code == 404:
                return None
                
            if not response.ok:
                self._log_response_and_raise(response)
            
            batch_data = response.json()
            
            # Convert API batch to Batch
            batch = Batch(
                id=str(batch_data['id']),
                qhit=qhit,
                track=batch_data['track'],
                stream=batch_data.get('additional_info', {}).get('stream'),
                timestamp=parser.isoparse(batch_data['timestamp'].replace("Z", "+00:00")).timestamp(),
                author=batch_data['author']
            )
            
            return batch
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def delete_batch(self, batch_id: str, q: Content | None = None) -> None:
        """
        Delete a batch and its associated tags
        """
        assert q is not None
        qhit = q.qid
        
        response = self.session.delete(
            f"{self.base_url}/{qhit}/batches/{batch_id}",
            headers=self._get_headers(q)
        )
        
        if not response.ok:
            self._log_response_and_raise(response)