from typing import Protocol

from src.tags.tagstore.types import Tag, UploadJob

class Tagstore(Protocol):
    def start_job(self,
        qhit: str,
        track: str,
        stream: str,
        author: str,
        auth: str | None=None,
    ) -> UploadJob:
        ...

    def upload_tags(self, tags: list[Tag], jobid: str, auth: str | None=None) -> None:
        ...

    def find_tags(self, auth: str | None=None, **filters) -> list[Tag]:
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

    def get_job(self, jobid: str, auth: str | None=None) -> UploadJob | None:
        ...

    def find_jobs(self, auth: str | None=None, **filters) -> list[str]:
        ...

    def count_tags(self, auth: str | None=None, **filters) -> int:
        ...