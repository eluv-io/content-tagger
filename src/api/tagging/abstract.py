from typing import Protocol

from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs, TagJobStatusReport, TagStartResult, TagStopResult

class TagAPI(Protocol):
    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        ...

    def status(self, qhit: str) -> list[TagJobStatusReport]:
        ...

    def stop(self, qhit: str, feature: str | None, stream: str | None) -> list[TagStopResult]:
        ...

    def shutdown_requested(self) -> bool:
        ...

    def cleanup(self) -> None:
        ...