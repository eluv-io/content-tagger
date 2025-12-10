from dataclasses import asdict
from datetime import datetime
import shutil
import json
import os
import base64
import time
import uuid

from src.common.content import Content
from src.tags.tagstore.model import *
from src.tags.tagstore.abstract import Tagstore

class FilesystemTagStore(Tagstore):
    def __init__(self, base_dir: str):
        self.base_path = base_dir
        os.makedirs(self.base_path, exist_ok=True)

    def create_track(self,
        qhit: str,
        name: str,
        label: str,
        q: Content | None = None,
    ) -> None:
        """
        Create a new track with metadata
        """
        track_dir = self._get_track_dir(qhit, name)
        os.makedirs(track_dir)

        track = Track(
            name=name,
            label=label,
            qhit=qhit
        )

        metadata_path = self._get_track_metadata_path(qhit, name)
        with open(metadata_path, 'w') as f:
            json.dump(asdict(track), f, indent=2)

    def get_track(self,
        qhit: str,
        name: str,
        q: Content | None = None,
    ) -> Track | None:
        """
        Get track metadata
        """
        metadata_path = self._get_track_metadata_path(qhit, name)
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                track_data = json.load(f)
                return Track(**track_data)
        except Exception:
            return None

    def create_batch(self,
        qhit: str,
        track: str,
        author: str,
        q: Content | None = None
    ) -> Batch:
        """
        Starts a new batch with provided metadata
        """
        batch_id = qhit + "/" + track + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[0:4]
        batch_dir = self._get_batch_dir(batch_id)
        os.makedirs(batch_dir, exist_ok=True)

        batch = Batch(
            id=batch_id,
            qhit=qhit,
            track=track,
            timestamp=time.time(),
            author=author
        )

        metadata_path = self._get_batch_metadata_path(batch_id)
        with open(metadata_path, 'w') as f:
            json.dump(asdict(batch), f, indent=2)

        return batch

    def upload_tags(self, 
        tags: list[Tag], 
        batch_id: str,
        q: Content | None = None
    ) -> None:
        """
        Upload tags for a specific batch, grouped by source
        """
        if not tags:
            return
        
        batch_dir = self._get_batch_dir(batch_id)
        if not os.path.exists(batch_dir):
            raise ValueError(f"Batch {batch_id} not found. Call start_batch() first.")
        
        # Group tags by source
        tags_by_source = {}
        for tag in tags:
            if tag.source not in tags_by_source:
                tags_by_source[tag.source] = []
            tags_by_source[tag.source].append(tag)
        
        # Write each source group to its own file
        for source, source_tags in tags_by_source.items():
            tags_path = self._get_tags_path(batch_id, source)
            
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

        NOTE: the filesystem implementation does not implement the shadowing logic.
        
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
        all_tags = []
        
        # First, get all batches that match batch-level filters
        batch_filters = {}
        if 'qhit' in filters:
            batch_filters['qhit'] = filters['qhit']
        if 'stream' in filters:
            batch_filters['stream'] = filters['stream']
        if 'track' in filters:
            batch_filters['track'] = filters['track']
        if 'author' in filters:
            batch_filters['author'] = filters['author']
        
        if 'batch_id' in filters:
            # If specific batch_id requested, only check that batch
            batch_ids = [filters['batch_id']]
        else:
            # Get all matching batches
            batch_ids = self.find_batches(**batch_filters)
        
        # Collect tags from matching batches
        for batch_id in batch_ids:
            tags = self._get_tags_for_batch(batch_id)
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
        batch_ids = []
        
        # Iterate through all directories in base_path
        if not os.path.exists(self.base_path):
            return batch_ids

        for batch_id, batch_dir in self._get_batch_ids_with_paths():

            # Skip if not a directory
            if not os.path.isdir(batch_dir):
                continue
            
            # Get batch metadata to check filters
            batch = self.get_batch(batch_id)
            if batch is None:
                continue
            
            # Apply filters
            if 'qhit' in filters and batch.qhit != filters['qhit']:
                continue
            if 'track' in filters and batch.track != filters['track']:
                continue
            if 'author' in filters and batch.author != filters['author']:
                continue
            if 'timestamp_gte' in filters and batch.timestamp < filters['timestamp_gte']:
                continue
            if 'timestamp_lte' in filters and batch.timestamp > filters['timestamp_lte']:
                continue
            
            batch_ids.append(batch_id)
        
        # Apply pagination
        if 'offset' in filters:
            offset = filters['offset']
            batch_ids = batch_ids[offset:]
        
        if 'limit' in filters:
            limit = filters['limit']
            batch_ids = batch_ids[:limit]
        
        return batch_ids

    def delete_batch(self, batch_id: str, q: Content | None = None) -> None:
        dir = self._get_batch_dir(batch_id)
        shutil.rmtree(dir, ignore_errors=True)

    def count_tags(self, q: Content | None = None, **filters) -> int:
        """Count tags matching the given filters without loading all data"""
        return len(self.find_tags(**filters))

    def count_batches(self, q: Content | None = None, **filters) -> int:
        """Count batches matching the given filters"""
        return len(self.find_batches(**filters))

    def get_batch(self, batch_id: str, q: Content | None=None) -> Batch | None:
        """
        Get batch metadata
        """
        metadata_path = self._get_batch_metadata_path(batch_id)
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            batch_data = json.load(f)
            return Batch(**batch_data)

    def _get_track_dir(self, qhit: str, track: str) -> str:
        """Get the directory path for a specific track"""
        return os.path.join(self.base_path, qhit, track)

    def _get_track_metadata_path(self, qhit: str, track: str) -> str:
        """Get the path to track metadata file"""
        return os.path.join(self._get_track_dir(qhit, track), "trackmeta.json")

    def _get_batch_ids_with_paths(self) -> list[tuple[str, str]]:
        """Get all batch IDs with their corresponding paths"""
        if not os.path.exists(self.base_path):
            return []
        
        # batch_ids are represented by qhit/track/batch_name
        batch_ids = []
        for qhit in os.listdir(self.base_path):
            qhit_path = os.path.join(self.base_path, qhit)
            if not os.path.isdir(qhit_path):
                continue
                
            for track in os.listdir(qhit_path):
                track_path = os.path.join(qhit_path, track)
                if not os.path.isdir(track_path):
                    continue
                    
                for batch_name in os.listdir(track_path):
                    batch_path = os.path.join(track_path, batch_name)
                    # Skip the trackmeta.json file
                    if not os.path.isdir(batch_path):
                        continue
                    batch_ids.append((f"{qhit}/{track}/{batch_name}", batch_path))

        return batch_ids

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

    def _get_batch_dir(self, batch_id: str) -> str:
        """Get the directory path for a specific batch"""
        return os.path.join(self.base_path, batch_id)

    def _get_batch_metadata_path(self, batch_id: str) -> str:
        """Get the path to batch metadata file"""
        return os.path.join(self._get_batch_dir(batch_id), "batchmetadata.json")

    def _get_tags_path(self, batch_id: str, source: str) -> str:
        """Get the path to tags file for a specific source"""
        encoded_source = self._encode_source_for_filename(source)
        return os.path.join(self._get_batch_dir(batch_id), f"{encoded_source}.json")

    def _get_tags_for_batch(self, batch_id: str) -> list[Tag]:
        """Helper method to get all tags for a specific batch without filtering"""
        all_tags = []
        batch_dir = self._get_batch_dir(batch_id)
        
        if not os.path.exists(batch_dir):
            return all_tags
        
        # Iterate through all tag files in the batch directory
        for filename in os.listdir(batch_dir):
            if filename.endswith('.json') and filename != 'batchmetadata.json':
                tags_path = os.path.join(batch_dir, filename)
                try:
                    with open(tags_path, 'r') as f:
                        tag_data = json.load(f)
                        tags = [Tag(**tag_dict) for tag_dict in tag_data]
                        all_tags.extend(tags)
                except Exception:
                    # Skip corrupted files
                    continue
        
        return all_tags