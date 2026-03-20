from typing import Protocol

from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs
from src.service.model import TagJobStatusResult, TagStopResult, TagStartResult, StatusArgs

class TaggerService(Protocol):
    """Service level interface for opertating the tagging. This is the interface the API will call."""
    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        ...

    def status(self, req: StatusArgs) -> list[TagJobStatusResult]:
        ...

    def stop(self, qid: str, feature: str | None) -> list[TagStopResult]:
        ...