import time

from src.common.content import Content
from src.common.errors import BadRequestError
from src.tagging.fabric_tagging.model import TagArgs
from src.service.model import *
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.service.abstract import TaggerService

def _tag_status_to_job_status(status: str) -> str:
    mapping: dict[str, str] = {
        "Fetching content": "running",
        "Tagging content": "running",
        "Completed": "succeeded",
        "Failed": "failed",
        "Stopped": "cancelled",
    }
    return mapping[status]

class DirectAPI(TaggerService):
    """Service implementation that sits on top of the tagger worker and directly calls the tagger functions with no job queue"""
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
    
    def status(self, req: StatusArgs) -> list[TagJobStatusResult]:
        if not req.qid:
            raise BadRequestError("qid parameter must be specified for direct api")
        
        res = self.tagger.status(req.qid)
        return [
            TagJobStatusResult(
                qid=req.qid,
                job_id=f"<{req.qid}, {r.model}, {r.stream}>",
                status=_tag_status_to_job_status(r.status.status),
                created_at=0,
                model=r.model,
                stream=r.stream,
                params={},
                tagger_details=TagDetails(
                    tag_status=r.status.status,
                    time_running=r.status.time_ended - r.status.time_started if r.status.time_ended else time.time() - r.status.time_started,
                    progress=(0.3 * len(r.status.downloaded_sources) + 0.7 * len(r.status.tagged_sources)) / len(r.status.total_sources),
                    tagging_progress=f"{len(r.status.tagged_sources)}/{len(r.status.downloaded_sources)}",
                    total_parts=len(r.status.total_sources),
                    downloaded_parts=len(r.status.downloaded_sources),
                    tagged_parts=len(r.status.tagged_sources),
                )
            ) for r in res
        ]

    def stop(self, qid: str, feature: str | None) -> list[TagStopResult]:
        res = self.tagger.stop(qid, feature)
        return [TagStopResult(
            job_id=str(r.job_id),
            message=r.message,
        ) for r in res]