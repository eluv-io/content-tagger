import os
import signal
import threading
from dataclasses import dataclass
import time

from src.common.content import Content
from src.common.logging import logger
from src.service.common import get_warning_response
from src.tagging.fabric_tagging.model import TagStatusResult, JobStateDescription
from src.tagging.fabric_tagging.tagger import TaggerWorker
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import *

logger = logger.bind(name="Tag Runner")

TERMINAL_STATUSES: set[JobStateDescription] = {"Completed", "Failed", "Stopped"}

def _job_status_from_report(report: TagStatusResult) -> job_status:
    mapping: dict[str, job_status] = {
        "Fetching content": "running",
        "Tagging content": "running",
        "Completed": "succeeded",
        "Failed": "failed",
        "Stopped": "cancelled",
    }
    status: job_status = mapping.get(report.status.status, "running")
    return status


@dataclass(frozen=True)
class TagRunnerConfig:
    poll_interval: float

@dataclass(frozen=True)
class JobInfo:
    id: str
    qid: str
    feature: str
    auth: str

class TagRunner:
    """Bridges the job queue and TaggerWorker.

    Periodically polls the JobStore for queued jobs, claims them, and runs them
    through TaggerWorker.  While jobs are running it polls for status updates
    and forwards them back to the queue, and checks for stop requests.
    """

    def __init__(
        self,
        tagger: TaggerWorker,
        jobstore: JobStore,
        cfg: TagRunnerConfig,
    ):
        self.tagger = tagger
        self.jobstore = jobstore
        self.cfg = cfg

        self._running_jobs: dict[str, JobInfo] = {}
        self._shutdown = threading.Event()
        self._quiescing = threading.Event()

    def start(self) -> None:
        """Start the background polling loops."""
        self._shutdown.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="tag-runner-poll")
        self._poll_thread.start()
        logger.info("TagRunner started")

    def quiesce(self) -> None:
        """Enter quiesce mode: stop accepting new jobs."""
        if self._quiescing.is_set():
            logger.info("Already quiescing")
            return
        n = len(self._running_jobs)
        logger.info(f"Quiesce requested — draining {n} running job(s) before exit")
        self._quiescing.set()

    def stop(self) -> None:
        """Signal both loops to stop and wait for them to finish."""
        self._shutdown.set()
        self._poll_thread.join()
        if not self._quiescing.is_set():
            # Hard stop: cancel any jobs still tracked
            for job in list(self._running_jobs.values()):
                try:
                    self.jobstore.update_job(
                        UpdateJobRequest(
                            id=job.id, 
                            status="cancelled",
                            error="tagger worker service was shut down or restarted"
                        ),
                        auth=job.auth,
                    )
                except Exception as e:
                    logger.opt(exception=e).warning("failed to cancel job on shutdown", job_id=job.id)
        self._running_jobs.clear()
        self.tagger.cleanup()

        logger.info("TagRunner stopped")

    def _poll_loop(self) -> None:
        """Periodically look for queued jobs, claim them, and kick off tagging."""
        while not self._shutdown.is_set():
            try:
                self._poll_once()
            except Exception as e:
                logger.opt(exception=e).error("error during job poll")
            try:
                self._status_tick()
            except Exception as e:
                logger.opt(exception=e).error("error during status tick")

            if self._quiescing.is_set() and not self._running_jobs:
                logger.info("Quiesce complete — all jobs drained, triggering shutdown")
                os.kill(os.getpid(), signal.SIGTERM)
                return

            self._shutdown.wait(self.cfg.poll_interval)

    def _poll_once(self) -> None:
        """Check for queued jobs and start them. Also checks for stop requests on running jobs."""

        if self._quiescing.is_set():
            # In quiesce mode only service stop requests — don't claim new work
            self._check_stop_requests()
            return

        # start queued jobs
        queued = self.jobstore.list_jobs(ListJobArgs(status="queued"), auth="")
        for item in queued:
            if item.id in self._running_jobs:
                continue

            if item.stop_requested:
                logger.info("skipping job with stop requested", job_id=item.id)
                self._set_stopped(item)
                continue

            claimed = self.jobstore.claim_job(item.id, item.auth)
            if not claimed:
                continue

            logger.info("claimed job", job_id=item.id, qid=item.qid)

            feature = item.params.feature
            self._running_jobs[item.id] = JobInfo(id=item.id, qid=item.qid, feature=feature, auth=item.auth)

            self._run_job(item)

        self._check_stop_requests()

    def _check_stop_requests(self) -> None:
        """Check for stop requests on running jobs."""
        stop_requested = self.jobstore.list_jobs(ListJobArgs(status="running"), auth="")
        for item in stop_requested:
            if not item.stop_requested:
                continue

            if item.id not in self._running_jobs:
                continue

            logger.info("stop requested for job", job_id=item.id)
            try:
                self.tagger.stop(item.qid, item.params.feature)
            except Exception as e:
                logger.opt(exception=e).warning("failed to stop job", job_id=item.id)

    def _run_job(self, item: QueueItem) -> None:
        log = logger.bind(job_id=item.id, qid=item.qid)
        try:
            content = Content(qid=item.qid, token=item.auth)
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
                    error=str(error),
                ),
                auth=item.auth,
            )
        except Exception as e:
            logger.opt(exception=e).warning("failed to update job with error status", job_id=item.id)
        finally:
            self._finish_job(item.id)

    def _status_tick(self) -> None:
        items = list(self._running_jobs.values())

        # group by qid so we make one status() call per content object
        qhits: dict[str, list[JobInfo]] = {}
        for item in items:
            qhits.setdefault(item.qid, []).append(item)

        for qid, job_items in qhits.items():
            try:
                reports = self.tagger.status(qid)
            except Exception:
                logger.opt(exception=True).warning("failed to get status", qid=qid)
                continue

            report_by_feature: dict[str, TagStatusResult] = {r.model: r for r in reports}

            for item in job_items:
                r = report_by_feature.get(item.feature)
                if r is None:
                    continue

                # push updated status back to the queue
                queue_status = _job_status_from_report(r)

                error = r.status.error

                details=TagDetails(
                    tag_status=r.status.status,
                    time_running=r.status.time_ended - r.status.time_started if r.status.time_ended else time.time() - r.status.time_started,
                    progress=(0.3 * len(r.status.downloaded_sources) + 0.7 * len(r.status.uploaded_sources)) / len(r.status.total_sources) if r.status.total_sources else 1.0,
                    tagging_progress=f"{len(r.status.uploaded_sources)}/{len(r.status.total_sources)}",
                    total_parts=len(r.status.total_sources),
                    downloaded_parts=len(r.status.downloaded_sources),
                    tagged_parts=len(r.status.tagged_sources),
                    warnings=get_warning_response(r.status.warnings) if r.status.warnings else None,
                )

                try:
                    self.jobstore.update_job(
                        UpdateJobRequest(
                            id=item.id, 
                            status=queue_status, 
                            status_details=details,
                            error=error,
                        ),
                        auth=item.auth,
                    )
                except Exception as e:
                    logger.opt(exception=e).warning("failed to update job", job_id=item.id)

                # if the job reached a terminal state, clean it up
                if r.status.status in TERMINAL_STATUSES:
                    self._finish_job(item.id)

    def _finish_job(self, id: str) -> None:
        self._running_jobs.pop(id, None)

    def _set_stopped(self, item: QueueItem) -> None:
        try:
            self.jobstore.update_job(
                UpdateJobRequest(
                    id=item.id, 
                    status="cancelled",
                ),
                auth=item.auth,
            )
        except Exception as e:
            logger.opt(exception=e).warning("failed to update job with stopped status", job_id=item.id)