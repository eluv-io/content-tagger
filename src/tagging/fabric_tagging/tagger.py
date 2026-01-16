import threading
from collections import defaultdict
import time
import queue
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any
from datetime import datetime
from uuid import uuid4 as uuid
import os
from src.common.logging.timing import timeit

from src.fetch.model import *
from src.tag_containers.containers import LiveTagContainer
from src.tag_containers.model import ContainerRequest
from src.tag_containers.registry import ContainerRegistry
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.common.content import Content
from src.common.errors import *
from src.fetch.factory import FetchFactory
from src.tags.tagstore.abstract import Tagstore
from src.tagging.fabric_tagging.message_types import *
from src.tagging.fabric_tagging.job_state import *
from src.tagging.fabric_tagging.model import *
from src.tagging.uploading.uploader import UploadSession

from src.common.logging import logger
logger = logger.bind(name="Fabric Tagger")

@dataclass
class Message:
    """Passed to the Fabric Tagger actor thread"""
    data: Request
    response_mailbox: queue.Queue

@dataclass
class Response:
    """Response from the Fabric Tagger actor thread"""
    data: Any
    error: Exception | None

class FabricTagger:
    """
    Handles the flow of downloading data from fabric, tagging, and uploading

    Manages a single actor thread which processes all requests sequentially.
    """

    def __init__(
            self, 
            system_tagger: ContainerScheduler,
            cregistry: ContainerRegistry,
            tagstore: Tagstore,
            fetcher: FetchFactory,
            cfg: FabricTaggerConfig
        ):

        self.system_tagger = system_tagger
        self.cregistry = cregistry
        self.tagstore = tagstore
        self.fetcher = fetcher
        self.cfg = cfg

        self.jobstore = JobStore()
        self.shutdown_requested = False

        self.mailbox = queue.Queue()
        
        self.actor_thread = threading.Thread(target=self._actor_loop, daemon=True)
        self.actor_thread.start()

        self._schedule_upload_tick()

    def tag(self, q: Content, args: TagArgs) -> str:
        request = TagRequest(q=q, args=args)
        return self._submit(request)

    def status(self, qhit: str) -> dict[str, dict[str, dict]]:
        request = StatusRequest(qhit=qhit)
        return self._submit(request)

    def stop(self, qhit: str, feature: str | None, stream: str | None) -> None:
        request = StopRequest(qhit=qhit, feature=feature, stream=stream, status="Stopped")
        return self._submit(request)

    def cleanup(self) -> None:
        request = CleanupRequest()
        return self._submit(request)

    def _end_job(self, jobid: JobID, status: Literal["Stopped", "Failed", "Completed"], error: Exception | None) -> None:
        """Used to abort a job early in case of error."""
        if error:
            logger.opt(exception=error).error("job ended with error", extra={"jobid": jobid, "status": status})
        request = StopRequest(qhit=jobid.qhit, feature=jobid.feature, stream=jobid.stream, status=status)
        return self._submit_async(request)

    def _submit(self, req: Request) -> Any:
        """synchronous request - adds a message to the mailbox and blocks till it gets a response"""
        if isinstance(req, EnterTaggingPhase):
            # avoid logging large data blobs
            log_req = asdict(req)
            log_req["data"] = f"<DownloadResult with {len(req.data.sources)} sources>"
        else:
            log_req = req
        logger.info("submitting synchronous request", extra={"request": log_req, "queue_size": self.mailbox.qsize()})
        if self.shutdown_requested:
            raise RuntimeError("FabricTagger received shutdown signal, cannot accept new requests")
        caller_mailbox = queue.Queue()
        message = Message(req, caller_mailbox)
        self.mailbox.put(message)
        response = caller_mailbox.get()
        if response.error:
            # actor stores the exception in the response so it can be safely re-raised here in the caller thread
            raise response.error
        return response.data

    def _submit_async(self, req: Request) -> None:
        """asynchronous request - adds a message to the mailbox and returns immediately."""
        if not isinstance(req, UploadTick):
            logger.info("submitting async request", extra={"request": req, "queue_size": self.mailbox.qsize()})
        message = Message(req, queue.Queue())
        self.mailbox.put(message)

    def _schedule_upload_tick(self):
        if not self.shutdown_requested:
            self.upload_timer = threading.Timer(0.2, lambda: self._submit_async(UploadTick()))
            self.upload_timer.start()

    def _actor_loop(self):
        """Main actor loop - processes all messages sequentially"""
        logger.info("FabricTagger actor started")

        while not self.shutdown_requested:
            try:
                message = self.mailbox.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._handle_message(message)
            except Exception as e:
                logger.opt(exception=e).error("error in actor loop", extra={"message": message.data})
                time.sleep(0.2)

        logger.info("FabricTagger actor shutting down")

    def _handle_message(self, message: Message):
        try:
            if isinstance(message.data, TagRequest):
                with timeit(f"handling tag request: {message.data}", min_duration=0.25):
                    self._handle_tag_request(message)
            elif isinstance(message.data, StatusRequest):
                self._handle_status_request(message)
            elif isinstance(message.data, StopRequest):
                self._handle_stop_request(message)
            elif isinstance(message.data, CleanupRequest):
                self._handle_cleanup_request(message)
            elif isinstance(message.data, UploadTick):
                with timeit(f"handling upload request: {message.data}", min_duration=0.5):
                    self._handle_upload_tick(message)
            elif isinstance(message.data, EnterFetchingPhase):
                with timeit(f"handling start fetch request: {message.data}", min_duration=0.25):
                    self._handle_enter_fetching_phase(message)
            elif isinstance(message.data, EnterTaggingPhase):
                with timeit(f"handling start tag request: {message.data}", min_duration=0.25):
                    self._handle_enter_tagging_phase(message)
            elif isinstance(message.data, EnterCompletePhase):
                self._handle_enter_complete_phase(message)
            else:
                logger.warning("received unknown message type", extra={"type": type(message.data), "message": message.data})
        except Exception as e:
            logger.opt(exception=e).error("error handling message", extra={"message": message.data})
            message.response_mailbox.put(Response(data=None, error=e))

    def _handle_tag_request(self, message: Message):
        assert isinstance(message.data, TagRequest)
        request: TagRequest = message.data

        q = request.q
        args = request.args
    
        self._validate_args(args)
        logger.info("processing tag request", extra={"qhit": q.qhit, "args": args})

        job = self._create_job(q, args.feature, args)

        jobid = job.get_id()
        if jobid in self.jobstore.active_jobs:
            message.response_mailbox.put(Response(data=f"{jobid} is already running", error=None))
        else:
            self.jobstore.active_jobs[jobid] = job

            self._submit_async(EnterFetchingPhase(job_id=jobid))

            message.response_mailbox.put(Response(data="Job started successfully", error=None))

    def _create_job(self, q: Content, feature: str, args: TagArgs) -> TagJob:
        """
        Initialize the starting state for a job and important context for its runtime
        """

        is_live = isinstance(args.scope, LiveScope)

        # for user to stop the job
        stop_event = threading.Event()

        worker = self.fetcher.get_session(
            q, 
            DownloadRequest(
                preserve_track=feature if not args.replace else "",
                output_dir=self._output_dir_from_q(q),
                scope=args.scope,
            ),
            stop_event
        )

        media_state = MediaState(
            downloaded=[],
            worker=worker
        )

        upload_session = UploadSession(
            tagstore=self.tagstore,
            feature=feature,
            source_q=q,
            destination_qid=args.destination_qid or q.qid,
            media_state=media_state,
            config=self.cfg.uploader
        )

        job = TagJob(
            state=JobState.starting(media_state, upload_session),
            args=JobArgs(**args.__dict__, q=q, retry_upload=is_live, retry_fetch=is_live),
            stop_event=stop_event,   
        )

        return job

    def _handle_enter_fetching_phase(self, message: Message):
        """Updates states and then starts fetching in background thread"""
        assert isinstance(message.data, EnterFetchingPhase)
        jobid = message.data.job_id

        if jobid not in self.jobstore.active_jobs:
            logger.warning(f"Received EnterFetchingPhase for inactive job: {jobid}")
            message.response_mailbox.put(Response(data=None, error=None))
            return

        job = self.jobstore.active_jobs[jobid]
        logger.info("entering fetching phase", extra={"jobid": jobid})
        job.state.status.status = "Fetching content"

        threading.Thread(
            target=self._do_fetching, 
            args=(job,), 
            daemon=True
        ).start()

        message.response_mailbox.put(Response(data=None, error=None))

    def _do_fetching(self, job: TagJob) -> None:
        """Perform fetching work, then request transition to tagging phase
        
        Runs in a background thread to avoid blocking the main thread.

        In the case of live jobs, if fetching fails, it will retry after a delay.
        """
        with timeit("fetching work", min_duration=5.5):
            jobid = job.get_id()

            try:
                dl_res = job.state.media.worker.download()
            except Exception as e:
                if job.args.retry_fetch:
                    retry_delay = 5
                    logger.error(f"Error during fetching but retry is set to true, retrying in {retry_delay} seconds\n{str(e)}", extra={"jobid": jobid})
                    threading.Timer(retry_delay, lambda: self._submit_async(EnterFetchingPhase(job_id=jobid))).start()
                else:
                    self._end_job(jobid, "Failed", e)
                return

            if job.stop_event.is_set():
                return

            self._submit_async(EnterTaggingPhase(job_id=jobid, data=dl_res))

    def _handle_enter_tagging_phase(self, message: Message):
        """Update state and start tagging in background thread"""
        assert isinstance(message.data, EnterTaggingPhase)
        jobid = message.data.job_id
        dl_res = message.data.data

        if jobid not in self.jobstore.active_jobs:
            raise TaggerRuntimeError("Job is inactive", jobid=jobid)

        job = self.jobstore.active_jobs[jobid]
        
        try:
            self._process_tagging_phase(job, dl_res)
            message.response_mailbox.put(Response(data=None, error=None))
        except Exception as e:
            self._cleanup_job(job, "Failed", "Tagging phase failed")
            raise TaggerRuntimeError("Tagging phase failed", jobid=jobid) from e

    def _process_tagging_phase(self, job: TagJob, dl_res: DownloadResult) -> None:
        """Process download results and update tagging state"""
        jobid = job.get_id()
        logger.info("entering tagging phase", extra={"jobid": jobid})

        job.state.status.status = "Tagging content"

        new_sources = []
        for s in dl_res.sources:
            if s.name not in job.state.media.downloaded:
                new_sources.append(s)
            else:
                logger.error("Got a duplicate source, ignoring", extra={"jobid": jobid, "source": s.name})

        job.state.media.downloaded += new_sources

        # mark any sources that failed to download
        for s in dl_res.failed:
            if s not in job.state.status.failed:
                job.state.status.failed.append(s)

        if not job.state.container and not new_sources and dl_res.done:
            logger.info("Fetcher finished with no media, aborting the job.", extra={"jobid": jobid})
            self._submit_async(EnterCompletePhase(job_id=jobid))
            return

        if new_sources:
            if not job.state.container:
                self._start_new_container(job)
            else:
                with timeit("sending new media to container", min_duration=0.2):
                    self._send_media(job.state.container, new_sources)

        if dl_res.done:
            # signal to container to stop accepting new media and shut down when done
            if job.state.container:
                job.state.container.send_eof()
        else:
            self._submit_async(EnterFetchingPhase(job_id=jobid))

    def _send_media(self, container: TagContainer, new_media: list[Source]) -> None:
        """
        Send the new media to the container for tagging via stdin
        """
        # right now this is a special class but in the future all containers will operate via stdin
        assert isinstance(container, LiveTagContainer)
        container.add_media([s.filepath for s in new_media])

    def _start_new_container(self, job: TagJob) -> None:
        media_input = [s.filepath for s in job.state.media.downloaded]
        container = self.cregistry.get(ContainerRequest(
            model_id=job.args.feature,
            media_input=media_input,
            run_config=job.args.run_config,
            live=isinstance(job.args.scope, LiveScope),
            job_id=job.args.q.qhit + "-" + datetime.now().strftime("%Y%m%d%H%M") + "-" + str(uuid())[0:6]
        ))
        
        uid = self.system_tagger.start(container, job.state.tagging_done)
        job.state.container = container
        job.state.taghandle = uid

        # dumb waiter thread, just waits on the event and then runs an upload
        threading.Thread(
            target=self._do_tagging, 
            args=(job,), 
            daemon=True
        ).start()

    def _do_tagging(self, job: TagJob) -> None:
        """Waits for tagging to complete, then uploads tags and requests transition to complete phase"""
        jobid = job.get_id()
        logger.info("waiting for tagging to complete", extra={"jobid": jobid})

        job.state.tagging_done.wait()

        if job.stop_event.is_set():
            logger.info("tagging was stopped via stop event", extra={"jobid": jobid})
            return

        status = self.system_tagger.status(job.state.taghandle)
        if status.status != "Completed":
            self._end_job(jobid, "Failed", RuntimeError(f"Tagging job ended with status: {status.status}"))
            return

        # Request transition to complete phase
        self._submit(EnterCompletePhase(job_id=jobid))

    def _handle_enter_complete_phase(self, message: Message):
        """Phase 3: Update state to complete (actor thread)"""
        assert isinstance(message.data, EnterCompletePhase)
        jobid = message.data.job_id

        if jobid not in self.jobstore.active_jobs:
            # TODO: test we should hit here if we stop a job during tagging
            logger.warning(f"Received EnterCompletePhase for inactive job: {jobid}")
            message.response_mailbox.put(Response(data=None, error=None))
            return

        job = self.jobstore.active_jobs[jobid]

        logger.info("entering complete phase", extra={"jobid": jobid})

        # catch any remaining tags - this blocks the main thread briefly but it's not called very often
        self._run_upload(job)

        for source in job.state.media.downloaded:
            if source.name not in job.state.upload_session.uploaded_sources:
                job.state.missing_tags.add(source.name)

        self._set_stop_state(jobid, "Completed", "")

        message.response_mailbox.put(Response(data=None, error=None))

    def _handle_status_request(self, message: Message):
        assert isinstance(message.data, StatusRequest)
        request: StatusRequest = message.data

        active_jobs = self.jobstore.active_jobs
        inactive_jobs = self.jobstore.inactive_jobs

        jobids = {jobid for jobid in active_jobs.keys() if jobid.qhit == request.qhit}
        jobids |= {jobid for jobid in inactive_jobs.keys() if jobid.qhit == request.qhit}

        if len(jobids) == 0:
            raise MissingResourceError(f"No jobs started for {request.qhit}")

        res = defaultdict(dict)
        for jid in jobids:
            job = active_jobs.get(jid) or inactive_jobs[jid]
            feature = job.args.feature
            # TODO: ugly logic
            stream = job.args.scope.stream if isinstance(job.args.scope, VideoScope | LiveScope) and job.args.scope.stream else "assets"
            res[stream][feature] = self._summarize_status(job.state)

        message.response_mailbox.put(Response(data=dict(res), error=None))

    def _handle_stop_request(self, message: Message):
        assert isinstance(message.data, StopRequest)
        request: StopRequest = message.data

        def _job_filter(jobid: JobID) -> bool:
            if jobid.qhit != request.qhit:
                return False
            if request.feature and jobid.feature != request.feature:
                return False
            if request.stream and jobid.stream != request.stream:
                return False
            return True

        active_jobs = self.jobstore.active_jobs
        jobids = {jobid for jobid in active_jobs.keys() 
                    if _job_filter(jobid)}

        if len(jobids) == 0:
            inactive_jobs = self.jobstore.inactive_jobs
            old_jobs = [jobid for jobid in inactive_jobs.keys() 
                        if _job_filter(jobid)]
            if old_jobs:
                raise MissingResourceError(f"Job for {request.feature} on {request.qhit} is already complete")
            raise MissingResourceError(f"No job running for {request.feature} on {request.qhit}")

        jobs = [active_jobs[jid] for jid in jobids]
        for job in jobs:
            self._set_stop_state(job.get_id(), request.status, "")

        # Stop system tagger jobs in background thread
        def stop_system_jobs():
            for job in jobs:
                if job.state.container is not None:
                    try:
                        self.system_tagger.stop(job.state.taghandle)
                    except Exception as e:
                        logger.opt(exception=e).error("error stopping job", extra={"jobid": job.get_id()})

        threading.Thread(target=stop_system_jobs, daemon=True).start()

        message.response_mailbox.put(Response(data=None, error=None))

    def _handle_cleanup_request(self, message: Message):
        logger.info("Shutting down fabric tagger...")
        
        # Stop all active jobs
        active_jobs = list(self.jobstore.active_jobs.keys())
        for job_id in active_jobs:
            self._set_stop_state(job_id, "Stopped", "Shutdown requested")
        
        self.shutdown_requested = True
        self.system_tagger.shutdown()

        message.response_mailbox.put(Response(data=None, error=None))

    def _handle_upload_tick(self, message: Message):
        assert isinstance(message.data, UploadTick)
        
        threading.Thread(
            target=self._upload_all_background,
            daemon=True
        ).start()
        
        message.response_mailbox.put(Response(data=None, error=None))

    def _upload_all_background(self) -> None:
        """Upload tags for all jobs in background thread"""
        try:
            self._upload_all()
        except Exception as e:
            logger.opt(exception=e).error("unexpected error in uploader")
        finally:
            self._schedule_upload_tick()

    def _upload_all(self) -> None:
        """Upload tags for all jobs"""
        jobs = list(self.jobstore.active_jobs.values())

        for job in jobs:
            self._run_upload(job)

    def _set_stop_state(
            self, 
            jobid: JobID, 
            status: JobStateDescription,
            message: str
        ) -> None:
        """Set job to stopped state (called from actor thread)
        
        Updates job state, and moves job to inactive jobs.

        Also sets stop event to signal any asynchronous operations to stop.
        """

        if status not in ("Completed", "Failed", "Stopped"):
            raise TaggerRuntimeError("Invalid stop status", jobid=jobid, status=status, msg=message)

        if jobid in self.jobstore.active_jobs:
            job = self.jobstore.active_jobs[jobid]

            job.state.status.status = status
            job.state.status.time_ended = time.time()
            job.state.message = message

            if job.stop_event:
                # for signalling background threads to stop
                job.stop_event.set()

            self.jobstore.inactive_jobs[jobid] = job
            del self.jobstore.active_jobs[jobid]
        else:
            logger.warning("Tried to stop an inactive job", extra={"jobid": jobid})

    def _summarize_status(self, state: JobState) -> dict:
        """Summarize job status (called from actor thread)
        
        Converts internal job state to a user-friendly dictionary.
        """
        status = deepcopy(state.status)
        end = time.time()
        if status.time_ended:
            end = status.time_ended
        total_sources = len(state.media.downloaded) if state.media else 0
        total_tagged = len(state.upload_session.uploaded_sources)
        res = {
            "status": status.status,
            "time_running": end - status.time_started,
            "tagging_progress": f"{total_tagged}/{total_sources}" if total_sources > 0 else "0/0",
            "missing_tags": list(state.missing_tags),
            "failed": status.failed,
        }
        if state.message:
            res["message"] = state.message
        return res

    def _run_upload(self, job: TagJob) -> None:
        """Run upload for a job"""
        try:
            self.__upload_tags(job)
        except Exception as e:
            # TODO: reaping the job might be a bit aggressive
            logger.opt(exception=e).error("error uploading tags for job", extra={"jobid": job.get_id()})
            self._cleanup_job(job, "Failed", "Tag upload failed")

    def _cleanup_job(
        self, 
        job: TagJob, 
        status: JobStateDescription,
        msg: str
    ) -> None:
        self._set_stop_state(job.get_id(), status, msg)

        def cleanup():
            try:
                self.system_tagger.stop(job.state.taghandle)
            except Exception as e:
                logger.opt(exception=e).error("error stopping job", extra={"jobid": job.get_id()})

        if job.state.taghandle is not None:
            # run in background thread to avoid blocking the main thread
            logger.info("cleaning up job in background thread due to failed upload", extra={"jobid": job.get_id()})
            threading.Thread(target=cleanup, daemon=True).start()

    def __upload_tags(self, job: TagJob) -> None:
        """Upload tags for a job"""
        if job.state.container is None:
            return

        with timeit("getting tags from container", min_duration=0.5):
            tags = job.state.container.tags()

        job.state.upload_session.upload_tags(tags, job.args.retry_upload)

    def _output_dir_from_q(self, q: Content) -> str:
        out = os.path.join(self.cfg.media_dir, q.qhit)
        os.makedirs(out, exist_ok=True)
        return out

    def _validate_args(self, args: TagArgs) -> None:
        """Validate args (called from actor thread)"""
        services = self.cregistry.services()
        modconfigs = self.cregistry.cfg.model_configs

        feature = args.feature
        if feature not in services:
            raise MissingResourceError(
                f"Invalid feature: {feature}. Available features: {', '.join(services)}"
            )

        if isinstance(args.scope, AssetScope):
            modeltype = modconfigs[feature].type

            if modeltype != "frame":
                raise MissingResourceError(
                    f"{feature} is not frame-level"
                )