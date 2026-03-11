import time

from src.common.content import Content
from src.common.errors import BadRequestError
from src.tagging.fabric_tagging.model import TagArgs
from src.service.model import *
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.service.abstract import TagAPI

def _tag_status_to_job_status(status: str) -> str:
    mapping: dict[str, str] = {
        "Fetching content": "running",
        "Tagging content": "running",
        "Completed": "succeeded",
        "Failed": "failed",
        "Stopped": "cancelled",
    }
    return mapping[status]

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
    
    def status(self, req: StatusArgs) -> list[TagJobStatusReport]:
        if not req.qid:
            raise BadRequestError("qid parameter must be specified for direct api")
        res = self.tagger.status(req.qid)
        return [TagJobStatusReport(
            qid=req.qid,
            job_id=str(r.job_id),
            status=_tag_status_to_job_status(r.status),
            message=r.message,
            created_at=time.time() - r.time_running,
            model=r.model,
            params={},
            tagger_details=TagDetails(
                tag_status=r.status,
                stream=r.stream,
                time_running=r.time_running,
                tagging_progress=r.tagging_progress,
                failed=r.failed,
            ),
        ) for r in res]

    def stop(self, qhit: str, feature: str | None, stream: str | None) -> list[TagStopResult]:
        res = self.tagger.stop(qhit, feature, stream)
        return [TagStopResult(
            job_id=str(r.job_id),
            message=r.message,
        ) for r in res]