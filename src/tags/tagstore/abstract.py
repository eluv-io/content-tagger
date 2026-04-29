from typing import Protocol

from src.common.content import Content
from src.tags.tagstore.model import *

class Tagstore(Protocol):
    def create_track(self, 
        name: str,
        label: str,
        q: Content
    ) -> None:
        ...

    def get_track(self,
        name: str,
        q: Content
    ) -> Track | None:
        ...

    def create_batch(self,
        track: str,
        author: str,
        q: Content
    ) -> Batch:
        ...
        
    def update_batch(self, 
        batch_id: str,
        additional_info: dict,
        q: Content
    ) -> None:
        ...

    def upload_tags(self, tags: list[Tag], batch_id: str, q: Content) -> None:
        ...

    def find_tags(self, q: Content, **filters) -> list[Tag]:
        """
        Find tags with flexible filtering.
        
        Supported filters:
        - qid: str
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
        ...

    def get_batch(self, batch_id: str, q: Content) -> Batch | None:
        ...

    def find_batches(self, q: Content, **filters) -> list[str]:
        ...

    def count_tags(self, q: Content, **filters) -> int:
        ...

    def delete_batch(self, batch_id: str, q: Content) -> None:
        ...