import threading
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
from src.tag_containers.containers import TagContainer
from src.tag_containers.model import ContainerRequest, Progress
from src.tag_containers.registry import ContainerRegistry
from src.tagging.fabric_tagging.source_resolver import SourceResolver
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.common.content import Content
from src.common.errors import *
from src.fetch.factory import FetchFactory
from src.tagging.uploading.align import adjust_progress_sources, align_tags
from src.tags.tagstore.abstract import Tagstore
from src.tagging.fabric_tagging.message_types import *
from src.tagging.fabric_tagging.job_state import *
from src.tagging.fabric_tagging.model import *
from src.tagging.uploading.uploader import UploadSession

from src.common.logging import logger
from src.tags.track_resolver import TrackResolver

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

    Spawns extra threads as needed for IO bound requests to avoid blocking main thread. 
    """

    def __init__(
        self, 
        system_tagger: ContainerScheduler,
        cregistry: ContainerRegistry,
        tagstore: Tagstore,
        fetcher: FetchFactory,
        track_resolver: TrackResolver,
        source_resolver: SourceResolver,
        cfg: FabricTaggerConfig,
    ):

        self.system_tagger = system_tagger
        self.cregistry = cregistry
        self.tagstore = tagstore
        self.fetcher = fetcher
        self.track_resolver = track_resolver
        self.source_resolver = source_resolver
        self.cfg = cfg

        self.jobstore = JobStore()
        self.shutdown_signal = False

        self.mailbox = queue.Queue()
        
        self.actor_thread = threading.Thread(target=self._actor_loop, daemon=True)
        self.actor_thread.start()

        self._upload_lock = threading.Lock()
        self._schedule_upload_tick()

    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        request = TagRequest(q=q, args=args)
        return self._submit(request)

    def status(self, qid: str) -> list[TagStatusResult]:
        request = StatusRequest(qid=qid)
        return self._submit(request)

    def stop(self, qid: str, feature: str | None) -> list[TagStopResult]:
        request = StopRequest(qid=qid, feature=feature, status="Stopped")
        return self._submit(request)

    def cleanup(self) -> None:
        request = CleanupRequest()
        return self._submit(request)

    def shutdown_requested(self) -> bool:
        return self.shutdown_signal
    
    def _submit(self, req: Request) -> Any:
        """synchronous request - adds a message to the mailbox and blocks till it gets a response"""
        logger.info("submitting synchronous request", extra={"request": req, "queue_size": self.mailbox.qsize()})
        if self.shutdown_requested():
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
        if not self.shutdown_requested():
            self.upload_timer = threading.Timer(0.2, lambda: self._submit_async(UploadTick()))
            self.upload_timer.start()

    def _actor_loop(self):
        """Main actor loop - processes all messages sequentially"""
        logger.info("FabricTagger actor started")

        while not self.shutdown_requested():
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
        logger.info("processing tag request", qid=q.qid, args=args)

        job = self._initialize_job(q, args.feature, args)

        # human readable id for the job (qid, feature, stream)
        jobid = job.get_id()
        if jobid in self.jobstore.active_jobs:
            message.response_mailbox.put(Response(data=TagStartResult(job_id=jobid, started=False, message=f"A job with params {jobid} is already running"), error=None))
        else:
            self.jobstore.active_jobs[jobid] = job

            self._submit_async(EnterFetchingPhase(job_id=jobid))

            message.response_mailbox.put(Response(data=TagStartResult(job_id=jobid, started=True, message="Job started successfully"), error=None))

    def _initialize_job(self, q: Content, feature: str, args: TagArgs) -> TagJob:
        """
        Initialize the starting state for a job and important context for its runtime
        """

        # TODO: redundant with the max_fetch_retries, consolidate
        is_live = isinstance(args.scope, LiveScope)

        # for user to stop the job
        stop_event = threading.Event()

        output_dir = self._output_dir_from_q(q)

        ignore_sources = []
        if not args.replace:
            ignore_sources = self.source_resolver.resolve(q, feature)

        worker = self.fetcher.get_session(
            q, 
            DownloadRequest(
                ignore_sources=ignore_sources,
                output_dir=output_dir,
                scope=args.scope,
            ),
            stop_event
        )

        media_state = MediaState(
            downloaded=[],
            worker=worker,
            output_dir=output_dir,
        )

        dest_q = Content(args.destination_qid or q.qid, q.token)

        upload_session = UploadSession(
            tagstore=self.tagstore,
            feature=feature,
            dest_q=dest_q,
            track_resolver=self.track_resolver,
            do_retry=is_live,
        )

        job = TagJob(
            state=JobState.starting(media_state, upload_session),
            args=JobArgs(
                **args.__dict__,
                q=q,
                retry_upload=is_live,
            ),
            stop_event=stop_event,   
        )

        return job

    def _handle_enter_fetching_phase(self, message: Message):
        """Updates states and then starts fetching in background thread"""
        assert isinstance(message.data, EnterFetchingPhase)
        jobid = message.data.job_id

        log = logger.bind(job_id=jobid)

        if jobid not in self.jobstore.active_jobs:
            log.warning("received EnterFetchingPhase for inactive job")
            message.response_mailbox.put(Response(data=None, error=None))
            return

        job = self.jobstore.active_jobs[jobid]
        log.info("entering fetching phase")
        job.state.status = "Fetching content"

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
            log = logger.bind(job_id=jobid)

            try:
                dl_res = job.state.media.worker.download()
                # reset on successful fetch
                job.state.fetch_retry_count = 0
            except Exception as e:
                if job.state.fetch_retry_count < job.args.max_fetch_retries:
                    job.state.fetch_retry_count += 1
                    retry_delay = 5
                    log.error(
                        "Error during fetching; retrying...", 
                        retry_count=job.state.fetch_retry_count, 
                        max_retries=job.args.max_fetch_retries
                    )
                    threading.Timer(
                        retry_delay,
                        lambda: self._submit_async(EnterFetchingPhase(job_id=jobid))
                    ).start()
                else:
                    self._request_job_end(jobid, "Failed", e)
                return

            if job.stop_event.is_set():
                return

            self._submit_async(EnterTaggingPhase(job_id=jobid, dl_result=dl_res))

    def _handle_enter_tagging_phase(self, message: Message):
        """Update state and start tagging in background thread"""
        assert isinstance(message.data, EnterTaggingPhase)
        jobid = message.data.job_id
        dl_result = message.data.dl_result

        if jobid not in self.jobstore.active_jobs:
            raise TaggerRuntimeError("Job is inactive", jobid=jobid)

        job = self.jobstore.active_jobs[jobid]

        try:
            self._process_tagging_phase(job, dl_result)
            message.response_mailbox.put(Response(data=None, error=None))
        except Exception as e:
            self._request_job_end(job.get_id(), "Failed", e)
            raise TaggerRuntimeError("Tagging phase failed", jobid=jobid) from e

    def _process_tagging_phase(self, job: TagJob, dl_res: DownloadResult) -> None:
        """Process download results and update tagging state"""
        jobid = job.get_id()
        log = logger.bind(job_id=jobid)
        log.info("entering tagging phase")

        new_sources = []
        for s in dl_res.sources:
            if s.name not in job.state.media.downloaded:
                new_sources.append(s)
            else:
                logger.warning("Got a duplicate source, ignoring", source=s.name)

        job.state.media.downloaded.extend(new_sources)

        for src in dl_res.failed:
            # TODO: should propagate real error here
            job.state.warnings.append(f"Failed to download {src}")

        job.state.status = "Tagging content"

        if not job.state.container and dl_res.done and not new_sources:
            # end early
            log.info("Fetcher finished with no media, aborting the job.")
            self._submit_async(EnterCompletePhase(job_id=jobid))
            return

        if new_sources:
            # TODO: should start the container at the start and then just add media only
            if not job.state.container:
                self._start_new_container(job)
            assert job.state.container is not None
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
        container.add_media([s.filepath for s in new_media])

    def _start_new_container(self, job: TagJob) -> None:
        media_dir = job.state.media.output_dir
        container = self.cregistry.get(ContainerRequest(
            model_id=job.args.feature,
            media_dir=media_dir,
            run_config=job.args.run_config,
            job_id=job.args.q.qid + "-" + datetime.now().strftime("%Y%m%d%H%M") + "-" + str(uuid())[0:6]
        ))
        
        uid = self.system_tagger.start(container, job.state.tagging_done)
        job.state.container = container
        job.state.taghandle = uid

        # dumb waiter thread, just waits on the event and then runs an upload when it's done
        threading.Thread(
            target=self._await_tagging, 
            args=(job,), 
            daemon=True
        ).start()

    def _await_tagging(self, job: TagJob) -> None:
        """Waits for tagging to complete, then uploads tags and requests transition to complete phase"""
        jobid = job.get_id()
        log = logger.bind(job_id=jobid)

        log.info("waiting for tagging to complete")

        job.state.tagging_done.wait()

        if job.stop_event.is_set():
            log.info("tagging was stopped via stop event")
            return

        status = self.system_tagger.status(job.state.taghandle)
        if status.status != "Completed":
            container_errors = []
            if job.state.container is not None:
                container_errors = job.state.container.errors()
            if container_errors:
                error = RuntimeError(container_errors[0].message)
            else:
                error = status.error or RuntimeError("Container exited unsuccessfully")
            self._request_job_end(jobid, "Failed", error=error)
            return

        # Request transition to complete phase
        self._submit(EnterCompletePhase(job_id=jobid))

    def _handle_enter_complete_phase(self, message: Message):
        """Phase 3: Update state to complete (actor thread)"""
        assert isinstance(message.data, EnterCompletePhase)
        jobid = message.data.job_id

        log = logger.bind(job_id=jobid)

        if jobid not in self.jobstore.active_jobs:
            # TODO: test we should hit here if we stop a job during tagging
            log.warning("Received EnterCompletePhase for inactive job")
            message.response_mailbox.put(Response(data=None, error=None))
            return

        job = self.jobstore.active_jobs[jobid]

        log.info("entering complete phase")

        # catch any remaining tags - this blocks the main thread briefly but it's not called very often
        # NOTE: this cannot error, so we will not get a stuck job
        self._run_upload(job)

        self._set_stop_state(job, "Completed", None)

        message.response_mailbox.put(Response(data=None, error=None))

    def _update_batches_with_report(self, job: TagJob, status: str) -> None:
        """Persist a status report into every batch the job created."""
        log = logger.bind(job_id=job.get_id())

        metadata = job.state.media.worker.metadata()
        
        all_sources = metadata.sources
        downloaded_sources = [s.name for s in job.state.media.downloaded]

        # TODO: container should always be non-None
        statuses = []
        if job.state.container:
            statuses = job.state.container.progress()

        tagged_sources = self._get_tagged_sources(job.state.media.downloaded, statuses)


        uploaded_sources = job.state.upload_session.get_uploaded_sources()
        
        upload_status = UploadStatus(
            all_sources=all_sources, 
            downloaded_sources=downloaded_sources, 
            tagged_sources=tagged_sources, 
            uploaded_sources=uploaded_sources
        )

        # Container info
        # TODO: again, container should always be non-None, we should enforce this invariant better
        container_info = None
        if job.state.container is not None:
            try:
                container_info = job.state.container.info()
            except Exception as e:
                log.opt(exception=e).warning("failed to get container info")
                container_info = ContainerInfo(image_name="", annotations={})
        else:
            container_info = ContainerInfo(image_name="", annotations={})


        tag_args = TagArgs(
            feature=job.args.feature,
            run_config=job.args.run_config,
            scope=job.args.scope,
            destination_qid=job.args.destination_qid,
            replace=job.args.replace,
            max_fetch_retries=job.args.max_fetch_retries,
        )

        assert job.state.time_ended is not None

        report = TagContentStatusReport(
            source_qid=job.args.q.qid,
            params=tag_args,
            container=container_info,
            upload_status=upload_status,
            job_status=JobRunStatus(
                status=job.state.status, 
                time_ran=time.strftime("%Hh %Mm %Ss", time.gmtime(job.state.time_ended - job.state.time_started))
            ),
        )
        
        job.state.upload_session.upload_report(report)

    def _handle_status_request(self, message: Message):
        assert isinstance(message.data, StatusRequest)
        request: StatusRequest = message.data

        active_jobs = self.jobstore.active_jobs
        inactive_jobs = self.jobstore.inactive_jobs

        jobids = {jobid for jobid in active_jobs.keys() if jobid.qid == request.qid}
        jobids |= {jobid for jobid in inactive_jobs.keys() if jobid.qid == request.qid}

        if len(jobids) == 0:
            raise MissingResourceError(f"No jobs started for {request.qid}")

        res = []
        for jid in jobids:
            job = active_jobs.get(jid) or inactive_jobs[jid]
            res.append(self._summarize_status(job))

        message.response_mailbox.put(Response(data=res, error=None))

    def _summarize_status(self, job: TagJob) -> TagStatusResult:
        state = job.state
        
        tagged_sources = []
        if state.container is not None:
            tagged_sources = state.container.progress()
            tagged_sources = self._get_tagged_sources(state.media.downloaded, tagged_sources)

        downloaded_sources = [s.name for s in state.media.downloaded]

        job_status = JobStatus(
            status=state.status,
            time_started=state.time_started,
            time_ended=state.time_ended,
            total_sources=state.media.worker.metadata().sources,
            downloaded_sources=downloaded_sources,
            tagged_sources=tagged_sources,
            uploaded_sources=state.upload_session.get_uploaded_sources(),
            warnings=state.warnings,
            # TODO: should be populated
            error=state.error
        )

        jobid = job.get_id()

        return TagStatusResult(
            status=job_status,
            model=jobid.feature,
            stream=jobid.stream,
        )

    def _handle_stop_request(self, message: Message):
        assert isinstance(message.data, StopRequest)
        request: StopRequest = message.data

        def _job_filter(jobid: JobID) -> bool:
            if jobid.qid != request.qid:
                return False
            if request.feature and jobid.feature != request.feature:
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
                raise MissingResourceError(f"Job for {request.feature} on {request.qid} is already complete")
            raise MissingResourceError(f"No job running for {request.feature} on {request.qid}")

        jobs = [active_jobs[jid] for jid in jobids]
        results = []
        for job in jobs:
            self._set_stop_state(job, request.status, request.error)
            results.append(TagStopResult(job_id=job.get_id(), message="stopping job"))

        message.response_mailbox.put(Response(data=results, error=None))

    def _handle_cleanup_request(self, message: Message):
        logger.info("Shutting down fabric tagger...")
        
        # Stop all active jobs
        active_jobs = list(self.jobstore.active_jobs.values())
        for job in active_jobs:
            self._set_stop_state(job, "Stopped", RuntimeError("Tagger worker was shutdown"))
        
        self.shutdown_signal = True
        self.system_tagger.cleanup()

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
        job: TagJob,
        status: Literal["Stopped", "Failed", "Completed"],
        error: Exception | None = None
    ) -> None:
        """Sets all the necessary state and performs cleanup for a stopped/failed/completed job
        """

        jobid = job.get_id()

        log = logger.bind(job_id=jobid)

        if jobid not in self.jobstore.active_jobs:
            log.warning("Trying to set stop state for inactive job")
            return
        
        if error:
            log.opt(exception=error).error(f"Job ended with error, setting status to {status}")

        job.state.error = str(error) if error else None

        job = self.jobstore.active_jobs[jobid]

        job.state.status = status
        job.state.time_ended = time.time()

        if job.stop_event:
            # for signalling background threads to stop
            job.stop_event.set()

        self._update_batches_with_report(job, status=status)

        self.jobstore.inactive_jobs[jobid] = job
        del self.jobstore.active_jobs[jobid]

        def cleanup():
            try:
                self.system_tagger.stop(job.state.taghandle)
            except Exception as e:
                log.opt(exception=e).error("error stopping job")

        if job.state.taghandle is not None:
            # run in background thread to avoid blocking the main thread
            threading.Thread(target=cleanup, daemon=True).start()

    def _request_job_end(
        self, 
        jobid: JobID, 
        status: Literal["Stopped", "Failed"], 
        error: Exception | None
    ) -> None:
        """Used to abort a job early in case of error.
        
        Asynchronously submits a StopRequest so that we go through the normal stop handler
        """
        request = StopRequest(
            qid=jobid.qid, 
            feature=jobid.feature, 
            status=status, 
            error=error
        )
        return self._submit_async(request)

    def _run_upload(self, job: TagJob) -> None:
        """Run upload for a job"""

        # we lock because _handle_enter_complete_phase and a thread spawned by _handle_upload_tick can run concurrently
        with self._upload_lock:
            try:
                self.__upload_tags(job)
            except Exception as e:
                self._request_job_end(job.get_id(), "Failed", e)

    def __upload_tags(self, job: TagJob) -> None:
        if job.state.container is None:
            return

        with timeit("getting tags from container", min_duration=0.5):
            tags = job.state.container.tags()

        stream_meta = job.state.media.worker.metadata()

        # TODO: align_tags also corrects the source name which is ugly
        aligned_tags = align_tags(tags, job.state.media.downloaded, fps=stream_meta.fps)

        statuses = job.state.container.progress()

        tagged_sources = adjust_progress_sources(statuses, job.state.media.downloaded)

        job.state.upload_session.upload_tags(aligned_tags, [p.source_media for p in tagged_sources])

    def _output_dir_from_q(self, q: Content) -> str:
        out = os.path.join(self.cfg.media_dir, q.qid)
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
            
    def _get_tagged_sources(self, sources: list[Source], statuses: list[Progress]) -> list[str]:
        """Get the list of sources that have generated tags based on container progress messages""" # deduplicate sources just in case, container progress can be noisy and send duplicates
        
        tagged_sources = []
        source_by_filepath = {s.filepath: s for s in sources}

        for status in statuses:
            src = source_by_filepath.get(status.source_media)
            if src is not None:
                tagged_sources.append(src.name)
            else:
                logger.warning("got progress update for unknown source media, ignoring", source_media=status.source_media)

        return list(set(tagged_sources))