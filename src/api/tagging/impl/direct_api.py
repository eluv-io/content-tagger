from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs, TagJobStatusReport, TagStartResult, TagStopResult
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.api.tagging.abstract import TagAPI

class DirectAPI(TagAPI):
    def __init__(self, tagger: FabricTagger):
        self.tagger = tagger

    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        return self.tagger.tag(q, args)

    def status(self, qhit: str) -> list[TagJobStatusReport]:
        return self.tagger.status(qhit)

    def stop(self, qhit: str, feature: str | None, stream: str | None) -> list[TagStopResult]:
        return self.tagger.stop(qhit, feature, stream)
    
    def cleanup(self) -> None:
        return self.tagger.cleanup()
    
    def shutdown_requested(self) -> bool:
        return self.tagger.shutdown_requested()