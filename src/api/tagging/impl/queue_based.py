from src.common.content import Content
from src.common.logging import logger
from src.fetch.model import AssetScope, LiveScope, TimeRangeScope, VideoScope
from src.tagging.fabric_tagging.model import (
    JobID,
    TagArgs,
    TagJobStatusReport,
    TagStartResult,
    TagStopResult,
)
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, JobStatus, ListJobArgs
from src.api.tagging.abstract import TagAPI

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
    """Drop-in replacement for FabricTagger that enqueues work via a JobStore
    instead of running it directly.

    Exposes the same public interface (tag / status / stop) so API handlers
    can use it without changes.
    """

    def __init__(self, jobstore: JobStore):
        self.jobstore = jobstore

    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        """Enqueue a tagging job and return immediately."""
        auth = q.token()
        stream = _stream_from_scope(args.scope)
        job_id = JobID(qhit=q.qhit, feature=args.feature, stream=stream)

        self.jobstore.create_job(
            CreateQueueItem(
                qid=q.qhit,
                params=args,
                status="queued",
                status_details=JobStatus(error=None, details=None),
            ),
            auth=auth,
        )

        logger.info("enqueued tagging job", job_id=str(job_id))
        return TagStartResult(started=True, job_id=job_id, message="Job enqueued")

    def status(self, qhit: str) -> list[TagJobStatusReport]:
        """Return the latest status for all jobs targeting *qhit*."""
        items = self.jobstore.list_jobs(ListJobArgs(qid=qhit), auth="")

        reports: list[TagJobStatusReport] = []
        for item in items:
            # if the TagRunner has already posted back a full report, use it
            if item.status_details.details is not None:
                reports.append(item.status_details.details)
            else:
                # job is still queued / no report yet — synthesise a minimal one
                stream = _stream_from_scope(item.params.scope)
                reports.append(
                    TagJobStatusReport(
                        job_id=JobID(qhit=item.qid, feature=item.params.feature, stream=stream),
                        status="Fetching content",
                        time_running=0,
                        tagging_progress="0/0",
                        missing_tags=[],
                        failed=[],
                        model=item.params.feature,
                        stream=stream,
                        message=item.status_details.error,
                    )
                )
        return reports

    def stop(self, qhit: str, feature: str | None, stream: str | None) -> list[TagStopResult]:
        """Request a stop for matching jobs in the queue."""
        items = self.jobstore.list_jobs(ListJobArgs(qid=qhit), auth="")

        results: list[TagStopResult] = []
        for item in items:
            if feature is not None and item.params.feature != feature:
                continue

            item_stream = _stream_from_scope(item.params.scope)
            if stream is not None and item_stream != stream:
                continue

            self.jobstore.stop_job(item.id, auth=item.auth)
            job_id = JobID(qhit=item.qid, feature=item.params.feature, stream=item_stream)
            results.append(TagStopResult(job_id=job_id, message="Stop requested"))
            logger.info("stop requested", job_id=str(job_id))

        return results
