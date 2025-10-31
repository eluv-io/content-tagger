from typing import Protocol

from src.common.content import Content
from src.tags.tagstore.model import Tag, UploadJob

class Tagstore(Protocol):
    def start_job(self,
        qhit: str,
        track: str,
        stream: str,
        author: str,
        q: Content | None=None,
    ) -> UploadJob:
        ...

    def upload_tags(self, tags: list[Tag], jobid: str, q: Content | None=None) -> None:
        ...

    def find_tags(self, q: Content | None=None, **filters) -> list[Tag]:
        """
        Find tags with flexible filtering.
        
        Supported filters:
        - qhit: str
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
        ...

    def get_job(self, jobid: str, q: Content | None=None) -> UploadJob | None:
        ...

    def find_jobs(self, q: Content | None=None, **filters) -> list[str]:
        ...

    def count_tags(self, q: Content | None=None, **filters) -> int:
        ...

    def delete_job(self, jobid: str, q: Content | None=None) -> None:
        ...