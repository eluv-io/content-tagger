from src.common.content import Content
from src.common.errors import MissingResourceError
from src.common.logging import logger
from src.fetch.model import AssetScope, LiveScope, TimeRangeScope, VideoScope
from src.service.model import (
    TagDetails,
    TagJobStatusReport,
    TagStartResult,
    TagStopResult,
)
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, JobStatus, ListJobArgs
from src.service.abstract import TagAPI

logger = logger.bind(name="Queue Client")


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


class QueueClient(TagAPI):

    def __init__(self, jobstore: JobStore):
        self.jobstore = jobstore

    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        """Enqueue a tagging job and return immediately."""
        auth = q.token()

        job = self.jobstore.create_job(
            CreateQueueItem(
                qid=q.qhit,
                params=args,
                status="queued",
                status_details=JobStatus(error=None, details=None),
            ),
            auth=auth,
        )

        logger.info("enqueued tagging job", job_id=str(job.id))
        return TagStartResult(started=True, created_at=job.created_at, job_id=job.id, message="Job enqueued")

    def status(self, qhit: str) -> list[TagJobStatusReport]:
        """Return the latest status for all jobs targeting *qhit*."""
        items = self.jobstore.list_jobs(ListJobArgs(qid=qhit), auth="")
        if not items:
            raise MissingResourceError(f"No tagging jobs found for qhit: {qhit}")

        reports: list[TagJobStatusReport] = []
        for item in items:
            # if the TagRunner has already posted back a full report, use it
            if item.status_details.details is not None:
                reports.append(TagJobStatusReport(
                    job_id=item.id,
                    status=item.status,
                    message=item.status_details.error,
                    created_at=item.created_at,
                    model=item.params.feature,
                    tagger_details=TagDetails(
                        tag_status=item.status,
                        stream=_stream_from_scope(item.params.scope),
                        time_running=item.status_details.details.time_running,
                        tagging_progress=item.status_details.details.tagging_progress,
                        failed=item.status_details.details.failed,
                    ),
                ))
            else:
                # job is still queued / no report yet — synthesise a minimal one
                stream = _stream_from_scope(item.params.scope)
                reports.append(
                    TagJobStatusReport(
                        job_id=item.id,
                        status=item.status,
                        created_at=item.created_at,
                        model=item.params.feature,
                        tagger_details=None,
                        message=None,
                    )
                )
        return reports

    def stop(self, qhit: str, feature: str | None, stream: str | None) -> list[TagStopResult]:
        """Request a stop for matching jobs in the queue."""
        items = self.jobstore.list_jobs(ListJobArgs(qid=qhit), auth="")
        items = [item for item in items if item.status in ("queued", "running")]
        items = [item for item in items if item.params.feature == feature or feature is None]

        if not items:
            errstr = f"No running jobs found for qhit: {qhit}"
            if feature:
                errstr += f", feature: {feature}"
            raise MissingResourceError(errstr)
        
        results: list[TagStopResult] = []
        for item in items:
            self.jobstore.stop_job(item.id, auth=item.auth)
            results.append(TagStopResult(job_id=item.id, message="Stop requested"))
            logger.info("stop requested", job_id=str(item.id))

        return results