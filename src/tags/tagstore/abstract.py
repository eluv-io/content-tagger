from typing import Protocol

from src.common.content import Content
from src.tags.tagstore.model import *

class Tagstore(Protocol):
    def create_track(self, 
        qid: str,
        name: str,
        label: str,
        q: Content | None=None,
    ) -> None:
        ...

    def get_track(self,
        qid: str,
        name: str,
        q: Content | None=None,
    ) -> Track | None:
        ...

    def create_batch(self,
        qid: str,
        track: str,
        author: str,
        q: Content | None=None,
    ) -> Batch:
        ...
        
    def update_batch(self, 
        qid: str,
        batch_id: str,
        additional_info: dict,
        q: Content | None=None,
    ) -> None:
        ...

    def upload_tags(self, tags: list[Tag], batch_id: str, q: Content | None=None) -> None:
        ...

    def find_tags(self, q: Content | None=None, **filters) -> list[Tag]:
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

    def get_batch(self, batch_id: str, q: Content | None=None) -> Batch | None:
        ...

    def find_batches(self, q: Content | None=None, **filters) -> list[str]:
        ...

    def count_tags(self, q: Content | None=None, **filters) -> int:
        ...

    def delete_batch(self, batch_id: str, q: Content | None=None) -> None:
        ...