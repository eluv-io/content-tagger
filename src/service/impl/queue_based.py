from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import asdict, replace as dc_replace
from functools import lru_cache
import time

from common_ml.utils.metrics import timeit

from src.api.arg_resolver import ArgsResolver
from src.api.tagging.request_format import JobSpec, StartJobsRequest
from src.common.content import Content, QAPIFactory
from src.common.errors import BadRequestError, JobConflictError, MissingResourceError
from src.common.logging import logger
from src.fetch.model import LiveScope
from src.service.dependency_resolver import DependencyResolver
from src.service.model import *
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

    def tag(self, q: Content, req: StartJobsRequest) -> list[TagStartResult]:
        """Submit jobs to the queue for tagging. 
        
        Resolves job dependencies and adds necessary job details to the job in the background before
        releasing the job to be picked up by a worker. 
        """
        self.arg_resolver.validate(req)
        results: list[TagStartResult] = []
        created: list[tuple[JobSpec, QueueItem]] = []
        for job in req.jobs:
            try:
                item = self.jobstore.create_job(CreateQueueItem(qid=q.qid, model=job.model), auth=q.token)
            except JobConflictError as e:
                results.append(TagStartResult(job_id="", started=False, created_at=time.time(), dependencies=[], message=str(e)))
                continue
            except Exception as e:
                # don't leave the jobs created so far pending, holding their models
                self._fail_pending(q, [item.id for _, item in created], e)
                raise
            created.append((job, item))
            results.append(TagStartResult(job_id=item.id, started=True, created_at=item.created_at, dependencies=[], message="Job submitted"))
        if created:
            job_ids = [item.id for _, item in created]
            req = dc_replace(req, jobs=[job for job, _ in created])
            self._executor.submit(contextvars.copy_context().run, self._resolve_and_release, q, job_ids, req)
        return results

    def _resolve_and_release(self, q: Content, job_ids: list[str], req: StartJobsRequest) -> None:
        """Resolve job information needed for tagging and releasing the jobs into 'queued' state."""
        try:
            with timeit("resolving tag args"):
                args = self.arg_resolver.resolve(req, q)
            for arg in args:
                logger.info("resolved tag args", qid=q.qid, model=arg.feature, args=arg)
            title = self._get_display_title(q)
            parents = self.dependency_resolver.resolve(q, job_ids, args)
            for job_id, arg, deps in zip(job_ids, args, parents):
                self.jobstore.release_job(
                    ReleaseJobRequest(id=job_id, params=arg, deps=deps, additional_info={"title": title}),
                    auth=q.token,
                )
        except Exception as e:
            if isinstance(e, BadRequestError):
                logger.warning(f"failed to release jobs: {e.message}", job_ids=job_ids)
            else:
                logger.opt(exception=e).error("failed to release jobs", job_ids=job_ids)
            self._fail_pending(q, job_ids, e)

    def _fail_pending(self, q: Content, job_ids: list[str], error: Exception) -> None:
        """Cleanup all jobs in the request in case of an error"""
        for job_id in job_ids:
            try:
                if self.jobstore.get_job(job_id).status == "pending":
                    self.jobstore.complete_job(CompleteJobRequest(id=job_id, status="failed", error=str(error)), auth=q.token)
            except Exception:
                logger.opt(exception=True).warning("failed to mark pending job as failed", job_id=job_id)

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
                tagger_details=item.status_details,
                tenant=item.tenant,
                user=item.user,
                title=item.additional_info.get("title", ""),
                error=item.error,
            ))
        return reports