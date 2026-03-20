from dataclasses import asdict
from time import time

from src.common.content import Content, QAPIFactory
from src.common.errors import BadRequestError, MissingResourceError
from src.common.logging import logger
from src.fetch.model import AssetScope, LiveScope, TimeRangeScope, VideoScope
from src.service.model import *
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs, QueueItem
from src.service.abstract import TaggerService

logger = logger.bind(name="Queue Service")

class QueueService(TaggerService):
    """
    Service implementation which sits in front of a job queue.

    This is intended to be used in production.
    """

    def __init__(
        self, 
        jobstore: JobStore,
        qfactory: QAPIFactory
    ):
        self.jobstore = jobstore
        self.qfactory = qfactory

    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        """Enqueue a tagging job and return immediately."""
        auth = q.token

        existing = self.jobstore.list_jobs(ListJobArgs(qid=q.qid), auth=auth)
        for item in existing:
            if item.status in ("queued", "running") and item.params.feature == args.feature:
                logger.info("duplicate job rejected", qid=q.qid, feature=args.feature, existing_job_id=str(item.id))
                return TagStartResult(
                    job_id="",
                    started=False,
                    message=f"A job with params {args} is already running",
                    created_at=item.created_at,
                )

        qapi = self.qfactory.create(q)

        title = qapi.content_object_metadata(metadata_subtree="/public/name")
        if not isinstance(title, str):
            raise BadRequestError(f"Received non-string value at /meta/public/name for qid={q.qid}")

        job = self.jobstore.create_job(
            CreateQueueItem(
                qid=q.qid,
                params=args,
                status="queued",
                status_details=None,
                additional_info={"title": title},
            ),
            auth=auth,
        )

        logger.info("enqueued tagging job", job_id=str(job.id), qid=q.qid)
        return TagStartResult(started=True, created_at=job.created_at, job_id=job.id, message="Job enqueued")

    def status(self, req: StatusArgs) -> list[TagJobStatusResult]:
        """Return the latest status for all jobs targeting *qid*."""
        items = self.jobstore.list_jobs(
            ListJobArgs(qid=req.qid, user=req.user, tenant=req.tenant), 
            auth="")
        if not items:
            raise MissingResourceError(f"No tagging jobs found for qid: {req.qid}")

        if req.title is not None:
            items = [item for item in items if item.additional_info.get("title") == req.title]
    
        return self._items_to_reports(items)

    def _items_to_reports(self, items: list[QueueItem]) -> list[TagJobStatusResult]:
        """Convert a list of QueueItems to TagJobStatusResult objects."""
        reports: list[TagJobStatusResult] = []
        for item in items:
            reports.append(TagJobStatusResult(
                qid=item.qid,
                job_id=item.id,
                status=item.status,
                created_at=item.created_at,
                model=item.params.feature,
                stream=_stream_from_scope(item.params.scope),
                params=asdict(item.params),
                tagger_details=item.status_details,
                tenant=item.tenant,
                user=item.user,
                title=item.additional_info.get("title", ""),
                error=item.error,
            ))
        return reports

    def stop(self, qid: str, feature: str | None) -> list[TagStopResult]:
        """Request a stop for matching jobs in the queue."""
        items = self.jobstore.list_jobs(ListJobArgs(qid=qid), auth="")
        items = [item for item in items if item.status in ("queued", "running")]
        items = [item for item in items if item.params.feature == feature or feature is None]

        if not items:
            errstr = f"No running jobs found for qid: {qid}"
            if feature:
                errstr += f", feature: {feature}"
            raise MissingResourceError(errstr)
        
        results: list[TagStopResult] = []
        for item in items:
            self.jobstore.stop_job(item.id, auth=item.auth)
            results.append(TagStopResult(job_id=item.id, message="Stop requested"))
            logger.info("stop requested", job_id=str(item.id))

        return results
    
def _stream_from_scope(scope) -> str:
    """Derive the stream identifier from a scope, matching TagJob.get_id() logic."""
    if isinstance(scope, AssetScope):
        return "assets"
    elif isinstance(scope, VideoScope):
        return scope.stream
    elif isinstance(scope, LiveScope):
        return "video"
    elif isinstance(scope, TimeRangeScope):
        return "video"
    else:
        raise ValueError(f"unknown scope type: {type(scope)}")