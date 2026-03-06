import threading
from dataclasses import dataclass

from src.common.content import ContentFactory
from src.common.logging import logger
from src.tagging.fabric_tagging.model import TagJobStatusReport, JobStateDescription
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import *

logger = logger.bind(name="Tag Runner")

TERMINAL_STATUSES: set[JobStateDescription] = {"Completed", "Failed", "Stopped"}

def _job_status_from_report(report: TagJobStatusReport) -> job_status:
    mapping: dict[str, job_status] = {
        "Fetching content": "running",
        "Tagging content": "running",
        "Completed": "succeeded",
        "Failed": "failed",
        "Stopped": "cancelled",
    }
    status: job_status = mapping.get(report.status, "running")
    return status


@dataclass(frozen=True)
class TagRunnerConfig:
    poll_interval: float = 5.0
    status_interval: float = 10.0

@dataclass(frozen=True)
class JobInfo:
    id: str
    qid: str
    feature: str
    auth: str

class TagRunner:
    """Bridges the job queue and FabricTagger.

    Periodically polls the JobStore for queued jobs, claims them, and runs them
    through FabricTagger.  While jobs are running it polls for status updates
    and forwards them back to the queue, and checks for stop requests.
    """

    def __init__(
        self,
        tagger: FabricTagger,
        jobstore: JobStore,
        content_factory: ContentFactory,
        cfg: TagRunnerConfig,
    ):
        self.tagger = tagger
        self.jobstore = jobstore
        self.content_factory = content_factory
        self.cfg = cfg

        self._running_jobs: dict[str, JobInfo] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

    def start(self) -> None:
        """Start the background polling loops."""
        self._shutdown.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="tag-runner-poll")
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True, name="tag-runner-status")
        self._poll_thread.start()
        self._status_thread.start()
        logger.info("TagRunner started")

    def stop(self) -> None:
        """Signal both loops to stop and wait for them to finish."""
        self._shutdown.set()
        self._poll_thread.join()
        self._status_thread.join()
        self.tagger.cleanup()
        logger.info("TagRunner stopped")

    def _poll_loop(self) -> None:
        """Periodically look for queued jobs, claim them, and kick off tagging."""
        while not self._shutdown.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.opt(exception=True).error("error during job poll")
            self._shutdown.wait(self.cfg.poll_interval)

    def _poll_once(self) -> None:
        """Check for queued jobs and start them. Also checks for stop requests on running jobs."""

        # start queued jobs
        queued = self.jobstore.list_jobs(ListJobArgs(status="queued"), auth="")
        for item in queued:
            with self._lock:
                if item.id in self._running_jobs:
                    continue

            claimed = self.jobstore.claim_job(item.id, item.auth)
            if not claimed:
                continue

            logger.info("claimed job", job_id=item.id, qid=item.qid)

            feature = item.params.feature
            with self._lock:
                self._running_jobs[item.id] = JobInfo(id=item.id, qid=item.qid, feature=feature, auth=item.auth)

            threading.Thread(
                target=self._run_job,
                args=(item,),
                daemon=True,
                name=f"tag-runner-job-{item.id}",
            ).start()

        # check for stop requests
        stop_requested = self.jobstore.list_jobs(ListJobArgs(status="running"), auth="")
        for item in stop_requested:
            if not item.stop_requested:
                continue

            with self._lock:
                if item.id not in self._running_jobs:
                    continue

            logger.info("stop requested for job", job_id=item.id)
            try:
                self.tagger.stop(item.qid, item.params.feature, stream=None)
            except Exception as e:
                logger.opt(exception=e).warning("failed to stop job", job_id=item.id)

    def _run_job(self, item: QueueItem) -> None:
        log = logger.bind(job_id=item.id, qid=item.qid)
        try:
            content = self.content_factory.create_content(item.qid, item.auth)
            result = self.tagger.tag(content, item.params)
            log.info("tag started", extra={"started": result.started, "message": result.message})
        except Exception as e:
            log.opt(exception=e).error("failed to start tagging job")
            self._error_job(item, e)

    def _error_job(self, item: QueueItem, error: Exception) -> None:
        try:
            self.jobstore.update_job(
                UpdateJobRequest(
                    id=item.id,
                    status="failed",
                    status_details=JobStatus(
                        error=str(error),
                        details=None
                    )
                ),
                auth=item.auth,
            )
        except Exception:
            logger.opt(exception=True).warning("failed to update job with error status", job_id=item.id)
        finally:
            self._finish_job(item.id)

    def _status_loop(self) -> None:
        """Periodically check the status of running jobs and sync with the queue."""
        while not self._shutdown.is_set():
            try:
                self._status_tick()
            except Exception:
                logger.opt(exception=True).error("error during status tick")
            self._shutdown.wait(self.cfg.status_interval)

    def _status_tick(self) -> None:
        with self._lock:
            items = list(self._running_jobs.values())

        # group by qhit so we make one status() call per content object
        qhits: dict[str, list[JobInfo]] = {}
        for item in items:
            qhits.setdefault(item.qid, []).append(item)

        for qhit, job_items in qhits.items():
            try:
                reports = self.tagger.status(qhit)
            except Exception:
                logger.opt(exception=True).warning("failed to get status", qhit=qhit)
                continue

            report_by_feature: dict[str, TagJobStatusReport] = {r.job_id.feature: r for r in reports}

            for item in job_items:
                report = report_by_feature.get(item.feature)
                if report is None:
                    continue

                # push updated status back to the queue
                queue_status = _job_status_from_report(report)
                try:
                    self.jobstore.update_job(
                        UpdateJobRequest(id=item.id, status=queue_status, status_details=JobStatus(
                            error=None,
                            details=report,
                        )),
                        auth=item.auth,
                    )
                except Exception:
                    logger.opt(exception=True).warning("failed to update job", job_id=item.id)

                # if the job reached a terminal state, clean it up
                if report.status in TERMINAL_STATUSES:
                    self._finish_job(item.id)

    def _finish_job(self, id: str) -> None:
        with self._lock:
            self._running_jobs.pop(id, None)