from typing import Protocol

from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs
from src.service.model import TagJobStatusReport, TagStopResult, TagStartResult, StatusArgs

class TagAPI(Protocol):
    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        ...

    def status(self, req: StatusArgs) -> list[TagJobStatusReport]:
        ...

    def stop(self, qid: str, feature: str | None) -> list[TagStopResult]:
        ...