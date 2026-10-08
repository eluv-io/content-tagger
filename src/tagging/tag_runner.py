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
    """Convert the tagger worker state into a valid job queue state"""
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
    """Tagger job bookeeping struct"""
    id: str
    qid: str
    feature: str
    stream: str
    auth: str

    def log_context(self) -> dict:
        return {"job_id": self.id, "qid": self.qid, "model": self.feature}

def _item_log_context(item: QueueItem) -> dict:
    # params should be resolved before reaching 'queued' state
    assert item.params is not None
    return {"job_id": item.id, "qid": item.qid, "model": item.params.feature}

class TagRunner:
    """The main worker loop - polls from the queue and runs tagging jobs.

    While jobs are running it polls for status updates and posts the status info 
    to the queue. It also handles user submitted stop requests and transitioning finished jobs to a terminal state.

    The quiesce feature allows the runner to stop gracefully: finishing all currently running jobs before shutting down.
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
        """Hard shutdown."""
        self._shutdown.set()
        self._poll_thread.join()
        if not self._quiescing.is_set():
            # Hard stop: end any jobs still tracked
            for job in list(self._running_jobs.values()):
                try:
                    self._complete(job, "failed", None, "tagger worker service was shut down or restarted")
                except Exception as e:
                    logger.opt(exception=e).warning("failed to end job on shutdown", job_id=job.id)
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
        """Claim the queued jobs the system has the resources for and start them."""
        if self._quiescing.is_set():
            return

        queued = self.jobstore.list_jobs(ListJobArgs(status="queued"), auth="")
        for item in queued:
            assert item.params is not None
            if item.id in self._running_jobs or not self.tagger.has_room(item.params.feature):
                continue
            with logger.contextualize(**_item_log_context(item)):
                self._try_start(item)

    def _try_start(self, item: QueueItem) -> None:
        claimed = self.jobstore.claim_job(item.id, item.auth)
        if not claimed:
            return

        logger.info("claimed job")

        # params should be resolved before reaching 'queued' state
        assert item.params is not None

        feature = item.params.feature
        stream = item.params.scope.get_stream()
        self._running_jobs[item.id] = JobInfo(id=item.id, qid=item.qid, feature=feature, stream=stream, auth=item.auth)

        self._run_job(item)

    def _run_job(self, item: QueueItem) -> None:
        # params should be resolved before reaching 'queued' state
        assert item.params is not None
        try:
            content = Content(qid=item.qid, token=item.auth)
            result = self.tagger.tag(content, item.params, job_id=item.id)
            logger.info("tag started", started=result.started, result=result.message)
        except Exception as e:
            logger.opt(exception=e).error("failed to start tagging job")
            self._error_job(item, e)

    def _error_job(self, item: QueueItem, error: Exception) -> None:
        try:
            self.jobstore.complete_job(CompleteJobRequest(id=item.id, status="failed", error=str(error)), auth=item.auth)
        except Exception as e:
            logger.opt(exception=e).warning("failed to update job with error status")
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

            reports_by_model_stream: dict[tuple[str, str], TagStatusResult] = {(r.model, r.stream) : r for r in reports}

            for item in job_items:
                with logger.contextualize(**item.log_context()):
                    self._report_status(item, reports_by_model_stream.get((item.feature, item.stream)))

    def _report_status(self, item: JobInfo, r: TagStatusResult | None) -> None:
        if r is None:
            return

        fetch_progress = len(r.status.downloaded_sources) / len(r.status.total_sources) if r.status.total_sources else 0
        
        if r.status.container_progress_ratio is None:
            # approximate with tagged parts
            tag_progress = len(r.status.uploaded_sources) / len(r.status.total_sources) if r.status.total_sources else 0
        else:
            tag_progress = r.status.container_progress_ratio

        details=TagDetails(
            tag_status=r.status.status,
            time_running=r.status.time_ended - r.status.time_started if r.status.time_ended else time.time() - r.status.time_started,
            progress=0.3 * fetch_progress + 0.7 * tag_progress,
            tagging_progress=f"{len(r.status.uploaded_sources)}/{len(r.status.total_sources)}",
            tagged_duration=r.status.tagged_duration,
            total_parts=len(r.status.total_sources),
            downloaded_parts=len(r.status.downloaded_sources),
            tagged_parts=len(r.status.tagged_sources),
            warnings=get_warning_response(r.status.warnings) if r.status.warnings else None,
        )

        try:
            if r.status.status in TERMINAL_STATUSES:
                self._complete(item, _job_status_from_report(r), details, r.status.error)
            else:
                job = self.jobstore.update_progress(item.id, details, auth=item.auth)
                if job.status == "cancelling":
                    self._stop(item)
        except Exception as e:
            logger.opt(exception=e).warning("failed to update job")

        # if the job reached a terminal state, clean it up
        if r.status.status in TERMINAL_STATUSES:
            self._finish_job(item.id)

    def _stop(self, item: JobInfo) -> None:
        logger.info("stop requested for job")
        try:
            self.tagger.stop(item.qid, item.feature)
        except Exception as e:
            logger.opt(exception=e).warning("failed to stop job")

    def _complete(self, item: JobInfo, status: job_status, details: TagDetails | None, error: str | None) -> None:
        req = CompleteJobRequest(id=item.id, status=status, status_details=details, error=error)
        if not self.jobstore.complete_job(req, auth=item.auth):
            logger.warning("queue refused to complete job", status=status)

    def _finish_job(self, id: str) -> None:
        self._running_jobs.pop(id, None)
