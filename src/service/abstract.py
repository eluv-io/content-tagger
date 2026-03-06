from typing import Protocol

from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs
from src.service.model import TagJobStatusReport, TagStopResult, TagStartResult

class TagAPI(Protocol):
    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        ...

    def status(self, qhit: str) -> list[TagJobStatusReport]:
        ...

    def stop(self, qhit: str, feature: str | None, stream: str | None) -> list[TagStopResult]:
        ...