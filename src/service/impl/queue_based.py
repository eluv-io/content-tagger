from concurrent.futures import ThreadPoolExecutor
import contextvars
import threading
from dataclasses import asdict
from functools import lru_cache

from common_ml.utils.metrics import timeit

from src.api.arg_resolver import ArgsResolver
from src.api.tagging.request_format import StartJobsRequest
from src.common.content import Content, QAPIFactory
from src.common.errors import BadRequestError, MissingResourceError
from src.common.logging import logger
from src.fetch.model import LiveScope
from src.service.dependency_resolver import DependencyResolver
from src.service.model import *
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import CompleteJobRequest, CreateQueueItem, ListJobArgs, QueueItem, ReleaseJobRequest
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
        dependency_resolver: DependencyResolver,
        arg_resolver: ArgsResolver,
        qfactory: QAPIFactory
    ):
        self.jobstore = jobstore
        self.dependency_resolver = dependency_resolver
        self.arg_resolver = arg_resolver
        self.qfactory = qfactory
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="job-release")
        # (qid, model) of requests whose jobs haven't been released yet
        self._pending: set[tuple[str, str]] = set()
        self._pending_lock = threading.Lock()

    def tag(self, q: Content, req: StartJobsRequest) -> list[TagStartResult]:
        """Write a pending job per requested job and return right away. The request is resolved in the background,
        after which the jobs are released to the queue, or failed if resolving raises. The whole request is refused if
        any of its models already has a pending request on the content."""
        self.arg_resolver.validate(req)
        keys = {(q.qid, job.model) for job in req.jobs}
        with self._pending_lock:
            overlap = keys & self._pending
            if overlap:
                raise BadRequestError(f"Jobs are already pending on {q.qid} for: {', '.join(sorted(m for _, m in overlap))}")
            self._pending |= keys

        try:
            items = [
                self.jobstore.create_job(CreateQueueItem(qid=q.qid, model=job.model), auth=q.token)
                for job in req.jobs
            ]
        except Exception:
            self._forget_pending(keys)
            raise
        job_ids = [item.id for item in items]
        self._executor.submit(contextvars.copy_context().run, self._release, q, job_ids, req)
        return [
            TagStartResult(job_id=item.id, started=True, created_at=item.created_at, dependencies=[], message="Job submitted")
            for item in items
        ]

    def _forget_pending(self, keys: set[tuple[str, str]]) -> None:
        with self._pending_lock:
            self._pending -= keys

    def _release(self, q: Content, job_ids: list[str], req: StartJobsRequest) -> None:
        try:
            with timeit("resolving tag args"):
                args = self.arg_resolver.resolve(req, q)
            for arg in args:
                logger.info("resolved tag args", qid=q.qid, model=arg.feature, args=arg)
            self.release(q, job_ids, args)
        except Exception as e:
            if isinstance(e, BadRequestError):
                logger.warning(f"failed to release jobs: {e.message}", job_ids=job_ids)
            else:
                logger.opt(exception=e).error("failed to release jobs", job_ids=job_ids)
            for job_id in job_ids:
                try:
                    if self.jobstore.get_job(job_id).status == "pending":
                        self.jobstore.complete_job(CompleteJobRequest(id=job_id, status="failed", error=str(e)), auth=q.token)
                except Exception:
                    logger.opt(exception=True).warning("failed to mark pending job as failed", job_id=job_id)
        finally:
            self._forget_pending({(q.qid, job.model) for job in req.jobs})

    def release(self, q: Content, job_ids: list[str], args: list[TagArgs]) -> None:
        """Release pending jobs (job_ids[i] runs args[i]) to the queue with their dependencies. A job duplicating one
        that is already queued or running is cancelled instead."""
        title = self._get_display_title(q)
        deps = self.dependency_resolver.resolve(q, job_ids, args)
        for job_id, arg, dep in zip(job_ids, args, deps):
            if dep.duplicate_of is not None:
                self.jobstore.cancel_job(job_id, auth=q.token, reason=f"Job {dep.duplicate_of} is already running for this model and stream")
                continue
            stream = arg.scope.get_stream()
            released = self.jobstore.release_job(
                ReleaseJobRequest(
                    id=job_id, 
                    params=arg, 
                    deps=dep.parents, 
                    additional_info={"title": title}, 
                    resource=f"{q.qid}/{arg.feature}/{stream}",
                ),
                auth=q.token,
            )
            if not released and self.jobstore.get_job(job_id).status == "pending":
                self.jobstore.cancel_job(job_id, auth=q.token, reason=f"A job is already active for {arg.feature} on stream {stream}")

    @lru_cache(maxsize=1024)
    def _get_display_title(self, q: Content) -> str:
        qapi = self.qfactory.create(q)
        title = qapi.content_object_metadata(metadata_subtree="/public/name")
        if not isinstance(title, str):
            return ""
        return title

    def status(self, req: StatusArgs) -> list[TagJobStatusResult]:
        """Return the latest status for all jobs targeting *qid*."""
        items = self.jobstore.list_jobs(
            ListJobArgs(qid=req.qid, user=req.user, tenant=req.tenant, include_unready=True), 
            auth="")
        if not items:
            raise MissingResourceError(f"No tagging jobs found for qid: {req.qid}")

        if req.title is not None:
            items = [item for item in items if req.title.lower() in item.additional_info.get("title", "").lower()]
    
        return self._items_to_reports(items)

    def stop(self, qid: str, feature: str | None) -> list[TagStopResult]:
        """Request a stop for matching jobs in the queue."""
        items = self.jobstore.list_jobs(ListJobArgs(qid=qid, include_unready=True), auth="")
        items = [item for item in items if item.status in ("pending", "queued", "running")]
        items = [item for item in items if item.model == feature or feature is None]

        if not items:
            errstr = f"No running jobs found for qid: {qid}"
            if feature:
                errstr += f", feature: {feature}"
            raise MissingResourceError(errstr)
        
        results: list[TagStopResult] = []
        for item in items:
            self.jobstore.cancel_job(item.id, auth=item.auth)
            results.append(TagStopResult(job_id=item.id, message="Stop requested"))
            logger.info("stop requested", job_id=str(item.id))

        return results
    
    def _items_to_reports(self, items: list[QueueItem]) -> list[TagJobStatusResult]:
        """Convert a list of QueueItems to TagJobStatusResult objects."""

        def report_status(status: str, is_live: bool) -> str:
            if is_live and status == "failed":
                return "cancelled"
            return status

        reports: list[TagJobStatusResult] = []
        for item in items:
            reports.append(TagJobStatusResult(
                qid=item.qid,
                job_id=item.id,
                status=report_status(item.status, item.params is not None and isinstance(item.params.scope, LiveScope)),
                created_at=item.created_at,
                model=item.model,
                stream=item.params.scope.get_stream() if item.params else "",
                params=asdict(item.params) if item.params else {},
                dependencies=item.deps,
                tagger_details=item.status_details,
                tenant=item.tenant,
                user=item.user,
                title=item.additional_info.get("title", ""),
                error=item.error,
            ))
        return reports