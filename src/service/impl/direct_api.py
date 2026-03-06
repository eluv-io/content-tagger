import time

from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs
from src.service.model import TagJobStatusReport, TagStopResult, TagStartResult
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.service.abstract import TagAPI

class DirectAPI(TagAPI):
    def __init__(self, tagger: FabricTagger):
        self.tagger = tagger

    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        res = self.tagger.tag(q, args)
        # convert to service level struct which has the same name
        return TagStartResult(
            job_id=str(res.job_id),
            started=res.started,
            created_at=time.time(),
            message=res.message,
        )

    def status(self, qhit: str) -> list[TagJobStatusReport]:
        res = self.tagger.status(qhit)
        return [TagJobStatusReport(
            job_id=str(r.job_id),
            status=r.status,
            message=r.message,
            time_running=r.time_running,
            tagging_progress=r.tagging_progress,
            created_at=time.time() - r.time_running,
            model=r.model,
            stream=r.stream,
            failed=r.failed,
        ) for r in res]

    def stop(self, qhit: str, feature: str | None, stream: str | None) -> list[TagStopResult]:
        res = self.tagger.stop(qhit, feature, stream)
        return [TagStopResult(
            job_id=str(r.job_id),
            message=r.message,
        ) for r in res]