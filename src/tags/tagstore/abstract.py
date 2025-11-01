from typing import Protocol

from src.common.content import Content
from src.tags.tagstore.model import Tag, Batch

class Tagstore(Protocol):
    def create_batch(self,
        qhit: str,
        track: str,
        stream: str,
        author: str,
        q: Content | None=None,
    ) -> Batch:
        ...

    def upload_tags(self, tags: list[Tag], batch_id: str, q: Content | None=None) -> None:
        ...

    def find_tags(self, q: Content | None=None, **filters) -> list[Tag]:
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
        ...

    def get_batch(self, batch_id: str, q: Content | None=None) -> Batch | None:
        ...

    def find_batches(self, q: Content | None=None, **filters) -> list[str]:
        ...

    def count_tags(self, q: Content | None=None, **filters) -> int:
        ...

    def delete_batch(self, batch_id: str, q: Content | None=None) -> None:
        ...