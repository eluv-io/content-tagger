import threading
from collections import defaultdict
import time
import math
import queue
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from datetime import datetime
from uuid import uuid4 as uuid

from loguru import logger

from src.tags.tagstore.types import Tag
from src.fetch.types import DownloadRequest
from src.tag_containers.types import ContainerRequest
from src.tag_containers.containers import ContainerRegistry
from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tagger.fabric_tagging.types import *
from src.common.content import Content
from src.common.errors import MissingResourceError
from src.fetch.fetch_video import Fetcher
from src.tags.tagstore.abstract import Tagstore

@dataclass
class TagRequest:
    q: Content
    args: TagArgs

    def __str__(self):
        return f"TagRequest(q={self.q}, args={self.args})"

@dataclass
class StatusRequest:
    qhit: str
    def __str__(self):
        return f"StatusRequest(qhit={self.qhit})"

@dataclass
class StopRequest:
    qhit: str
    feature: str | None
    stream: str | None
    status: Literal["Stopped", "Failed", "Completed"]

    def __str__(self):
        return f"StopRequest(qhit={self.qhit}, feature={self.feature}, stream={self.stream}, status={self.status})"

@dataclass
class JobTransition:
    job_id: JobID
    data: Any

    def __str__(self):
        return f"JobTransition(job_id={self.job_id}, data={self.data})"

@dataclass
class UploadTick:
    def __str__(self):
        return "UploadTick()"

@dataclass
class CleanupRequest:
    def __str__(self):
        return "CleanupRequest()"

Request = TagRequest | StatusRequest | StopRequest | JobTransition | CleanupRequest | UploadTick

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
            system_tagger: SystemTagger,
            cregistry: ContainerRegistry,
            tagstore: Tagstore,
            fetcher: Fetcher
        ):

        self.system_tagger = system_tagger
        self.cregistry = cregistry
        self.tagstore = tagstore
        self.fetcher = fetcher

        self.jobstore = JobStore()
        self.shutdown_requested = False
        
        self.mailbox = queue.Queue()
        
        self.actor_thread = threading.Thread(target=self._actor_loop, daemon=True)
        self.actor_thread.start()

        self._schedule_upload_tick()

    def tag(self, q: Content, args: TagArgs) -> dict:
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
        if error:
            logger.opt(exception=error).error(f"Job {jobid} ended with error")
        request = StopRequest(qhit=jobid.qhit, feature=jobid.feature, stream=jobid.stream, status=status)
        return self._submit(request)

    def _submit(self, req: Request) -> Any:
        if self.shutdown_requested:
            raise RuntimeError("FabricTagger received shutdown signal, cannot accept new requests")
        caller_mailbox = queue.Queue()
        message = Message(req, caller_mailbox)
        self.mailbox.put(message)
        response = caller_mailbox.get()
        if response.error:
            raise response.error
        return response.data

    def _submit_async(self, req: Request) -> None:
        message = Message(req, queue.Queue())
        self.mailbox.put(message)

    def _schedule_upload_tick(self):
        """Schedule periodic upload tick"""
        if not self.shutdown_requested:
            self.upload_timer = threading.Timer(0.2, lambda: self._submit_async(UploadTick()))
            self.upload_timer.start()

    def _actor_loop(self):
        """Main actor loop - processes all messages sequentially"""
        logger.info("FabricTagger actor started")
        
        while not self.shutdown_requested:
            try:
                message = self.mailbox.get(timeout=1.0)
                self._handle_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Error in actor loop: {e}")
                time.sleep(0.2)
        
        logger.info("FabricTagger actor shutting down")

    def _handle_message(self, message: Message):
        """Handle a single message"""
        try:
            if isinstance(message.data, TagRequest):
                self._handle_tag_request(message)
            elif isinstance(message.data, StatusRequest):
                self._handle_status_request(message)
            elif isinstance(message.data, StopRequest):
                self._handle_stop_request(message)
            elif isinstance(message.data, CleanupRequest):
                self._handle_cleanup_request(message)
            elif isinstance(message.data, UploadTick):
                # called internally from upload timer
                self._handle_upload_tick(message)
            elif isinstance(message.data, JobTransition):
                # called internally from tagger thread
                self._handle_job_transition(message)
            else:
                logger.warning(f"Unknown message type: {type(message.data)}")
        except Exception as e:
            logger.exception(f"Error handling message {message.data}: {e}")
            # catch all errors to avoid crashing the actor thread, passed to caller who can raise
            message.response_mailbox.put(Response(data=None, error=e))

    def _handle_tag_request(self, message: Message):
        assert isinstance(message.data, TagRequest)
        request: TagRequest = message.data
    
        self._validate_args(request.args)
        args = self._assign_default_streams(request.args)
        logger.debug(args)
        
        status = {}
        for feature in args.features:
            stream = args.features[feature].stream
            assert stream is not None
            tsjob = self.tagstore.start_job(
                qhit=request.q.qhit,
                track=feature,
                stream=stream,
                author="tagger",
                auth=request.q._client.token
            )

            job = TagJob(
                state=JobState.starting(),
                args=JobArgs(
                    q=request.q,
                    feature=feature,
                    replace=args.replace,
                    runconfig=args.features[feature],
                    scope=args.scope
                ),
                upload_job=tsjob.id,
                stop_event=threading.Event(),
                tagging_done=None,
            )

            status[feature] = self._start_job(job)

        message.response_mailbox.put(Response(data=status, error=None))

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
            stream, feature = job.args.runconfig.stream, job.args.feature
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
            self._set_stop_state(job.get_id(), request.status, "", None)

        # Stop system tagger jobs in background thread
        def stop_system_jobs():
            for job in jobs:
                if not job.state.status == "Taggging content":
                    continue
                try:
                    self.system_tagger.stop(job.state.taghandle)
                except Exception as e:
                    logger.error(f"Error stopping job {job.get_id()}: {e}")

        if any(job.state.status == "Tagging content" for job in jobs):
            threading.Thread(target=stop_system_jobs, daemon=True).start()

        message.response_mailbox.put(Response(data=None, error=None))

    def _handle_cleanup_request(self, message: Message):
        logger.info("Shutting down fabric tagger...")
        
        # Stop all active jobs
        active_jobs = list(self.jobstore.active_jobs.keys())
        for job_id in active_jobs:
            self._set_stop_state(job_id, "Stopped", "Shutdown requested", None)
        
        self.shutdown_requested = True
        self.system_tagger.shutdown()

        message.response_mailbox.put(Response(data=None, error=None))

    def _handle_upload_tick(self, message: Message):
        assert isinstance(message.data, UploadTick)
        try:
            self._upload_all()
        except Exception as e:
            logger.exception(f"Unexpected error in uploader: {e}")
        self._schedule_upload_tick()
        message.response_mailbox.put(Response(data=None, error=None))

    def _handle_job_transition(self, message: Message):
        assert isinstance(message.data, JobTransition)
        update: JobTransition = message.data

        if update.job_id not in self.jobstore.active_jobs:
            raise ValueError(f"Received update for inactive job: {update.job_id}")

        job = self.jobstore.active_jobs[update.job_id]

        job_status = job.state.status.status

        if job_status not in ("Starting", "Fetching content", "Tagging content", "Uploading tags"):
            raise ValueError(f"{job} does not have a next state.")
        
        job = self.jobstore.active_jobs[update.job_id]
        
        try:
            if job_status == "Starting":
                self._enter_fetching_stage(job, update.data)
            elif job_status == "Fetching content":
                self._enter_tagging_stage(job, update.data)
            elif job_status == "Tagging content":
                self._enter_upload_stage(job, update.data)
            elif job_status == "Uploading tags":
                self._enter_complete_stage(job, update.data)
        except Exception as e:
            self._set_stop_state(job.get_id(), "Failed", "", e)

        message.response_mailbox.put(Response(data=None, error=None))

    def _enter_complete_stage(self, job: TagJob, data: Any) -> None:
        self._set_stop_state(job.get_id(), "Completed", "", None)
        assert job.state.media is not None
        for source in job.state.media.successful_sources:
            if source.name not in job.state.uploaded_sources:
                job.state.status.failed.append(source.name)

    def _enter_fetching_stage(self, job: TagJob, data: Any) -> None:
        job.state.status.status = "Fetching content"

    def _enter_tagging_stage(self, job: TagJob, data: Any) -> None:
        assert isinstance(data, DownloadResult)
        job.state.status.status = "Tagging content"
        job.state.media = data
        job.state.status.failed += data.failed
        media_files = [s.filepath for s in data.successful_sources]
        container = self.cregistry.get(ContainerRequest(
            model=job.args.feature,
            file_args=media_files,
            run_config=job.args.runconfig.model,
            job_id=job.args.q.qhit + "-" + datetime.now().strftime("%Y%m%d%H%M") + "-" + str(uuid())[0:6]
        ))
        reqresources = self.cregistry.get_model_config(job.args.feature).resources
        tagging_done = threading.Event()
        uid = self.system_tagger.start(container, reqresources, tagging_done)
        job.state.container = container
        job.state.taghandle = uid
        job.tagging_done = tagging_done

    def _enter_upload_stage(self, job: TagJob, data: Any) -> None:
        job.state.status.status = "Uploading tags"
        self._run_upload(job)

    def _start_job(self, job: TagJob) -> str:
        """
        Checks if job is startable, if so sends an async message to start it and returns
        """

        jobid = job.get_id()

        if jobid in self.jobstore.active_jobs:
            return f"{jobid} is already running"

        self.jobstore.active_jobs[jobid] = job

        threading.Thread(target=self._run_job, args=(job,), daemon=True).start()

        return "Job started successfully"

    def _run_job(self, job: TagJob) -> None:
        logger.debug(job.args)

        jobid = job.get_id()

        # 1. start fetching
        assert job.state.status.status == "Starting"
        self._submit(JobTransition(job_id=jobid, data=None))

        try:
            stream = job.args.runconfig.stream
            assert stream is not None
            dl_res = self.fetcher.download(
                job.args.q, 
                DownloadRequest(
                    stream_name=stream,
                    scope=job.args.scope,
                    preserve_track=job.args.feature if not job.args.replace else "",
                ),
                exit_event=job.stop_event
            )
            if job.stop_event.is_set():
                return
        except MissingResourceError as e:
            # str(e)
            self._end_job(jobid, "Failed", e)
            return
        except Exception as e:
            self._end_job(jobid, "Failed", e)
            return

        if len(dl_res.successful_sources) == 0:
            # Nothing left to tag
            self._end_job(jobid, "Completed", None)
            return

        # start tagging
        assert job.state.status.status == "Fetching content"
        self._submit(JobTransition(job_id=jobid, data=dl_res))

        assert job.tagging_done is not None
        job.tagging_done.wait()

        if job.stop_event.is_set():
            return

        status = self.system_tagger.status(job.state.taghandle)
        if status.status != "Completed":
            self._end_job(jobid, "Failed", RuntimeError(f"Tagging job ended with status: {status.status}"))
            return

        # start upload
        assert job.state.status.status == "Tagging content"
        self._submit(JobTransition(job_id=jobid, data=None))

        if job.stop_event.is_set():
            return

        # Set job to completed
        assert job.state.status.status == "Uploading tags"
        self._submit(JobTransition(job_id=jobid, data=None))

    def _set_stop_state(
            self, 
            jobid: JobID, 
            status: JobStateDescription,
            message: str,
            error: Exception | None,
        ) -> None:
        """Set job to stopped state (called from actor thread)
        
        Updates job state, and moves job to inactive jobs.

        Also sets stop event to signal any asynchronous operations to stop.
        """

        if error:
            logger.exception(error)

        if status not in ("Completed", "Failed", "Stopped"):
            raise ValueError(f"Invalid stop status: {status}")

        if jobid in self.jobstore.active_jobs:
            job = self.jobstore.active_jobs[jobid]

            job.state.status.status = status
            job.state.status.time_ended = time.time()
            job.state.message = message

            if job.stop_event:
                job.stop_event.set()

            self.jobstore.inactive_jobs[jobid] = job
            del self.jobstore.active_jobs[jobid]
        else:
            logger.warning(f"Tried to stop an inactive job: {jobid}")

    def _summarize_status(self, state: JobState) -> dict:
        """Summarize job status (called from actor thread)
        
        Converts internal job state to a user-friendly dictionary.
        """
        status = deepcopy(state.status)
        end = time.time()
        if status.time_ended:
            end = status.time_ended
        total_sources = len(state.media.successful_sources) if state.media else 0
        total_tagged = len(state.uploaded_sources)
        res = {
            "status": status.status,
            "time_running": end - status.time_started,
            "tagging_progress": f"{int(total_tagged / total_sources * 100)}%" if total_sources > 0 else "0%",
            "failed": status.failed,
        }
        if state.message:
            res["message"] = state.message
        return res

    def _upload_all(self) -> None:
        """Upload tags for all jobs (called from actor thread)"""
        jobs = list(self.jobstore.active_jobs.values())
        jobs += list(self.jobstore.inactive_jobs.values())
        
        for job in jobs:
            try:
                self._run_upload(job)
            except Exception as e:
                self._set_stop_state(job.get_id(), "Failed", "Tag upload failed", e)

    def _run_upload(self, job: TagJob) -> None:
        """Run upload for a job (called from uploader thread)"""
        try:
            self.__upload_tags(job)
        except Exception as e:
            self._cleanup_job(job, "Failed", "Tag upload failed", e)

    def _cleanup_job(
            self, 
            job: TagJob, 
            status: JobStateDescription,
            msg: str,
            error: Exception | None,
        ) -> None:
        self._set_stop_state(job.get_id(), status, msg, error)

        def cleanup():
            try:
                self.system_tagger.stop(job.state.taghandle)
            except Exception as e:
                logger.error(f"Error stopping job {job.get_id()}: {e}")

        if job.state.status.status == "Tagging content" and job.state.taghandle is not None:
            threading.Thread(target=cleanup, daemon=True).start()

    def __upload_tags(self, job: TagJob) -> None:
        """Upload tags for a job (called from actor thread)"""
        if job.state.status.status not in ("Uploading tags", "Tagging content"):
            return
            
        assert job.state.media is not None
        media_to_source = {s.filepath: s for s in job.state.media.successful_sources}

        assert job.state.container is not None
        outputs = job.state.container.tags()
        new_outputs = [out for out in outputs if media_to_source[out.source_media].name not in job.state.uploaded_sources]

        if not new_outputs:
            return

        stream_meta = job.state.media.stream_meta
        tags2upload = []
        for out in new_outputs:
            original_src = media_to_source[out.source_media]
            for tag in out.tags:
                tag.source = original_src.name
                tag.jobid = job.upload_job
                tags2upload.append(self._fix_tag_offsets(tag, original_src.offset, stream_meta.fps if stream_meta else None))

        self.tagstore.upload_tags(tags2upload, job.upload_job, auth=job.args.q._client.token)

        job.state.uploaded_sources.extend(media_to_source[out.source_media].name for out in new_outputs)

    def _validate_args(self, args: TagArgs) -> None:
        """Validate args (called from actor thread)"""
        services = self.cregistry.services()
        modconfigs = self.cregistry.cfg.model_configs

        for feature in args.features:
            if feature not in services:
                raise MissingResourceError(
                    f"Invalid feature: {feature}. Available features: {', '.join(services)}"
                )

            if isinstance(args, TagArgs):
                continue

            modeltype = modconfigs[feature].type

            if modeltype != "frame":
                raise MissingResourceError(
                    f"{feature} is not frame-level"
                )

    def _fix_tag_offsets(self, tag: Tag, offset: float, fps: float | None) -> Tag:
        """Fix tag offsets (called from actor thread)"""
        if tag.start_time is not None:
            tag.start_time += int(offset * 1000)
        if tag.end_time is not None:
            tag.end_time += int(offset * 1000)
        if "frame_tags" in tag.additional_info:
            if fps is not None:
                tag = self._fix_frame_indices(tag, offset, fps)
            else:
                logger.warning(f"Audio stream has frame_tags, removing them: {tag.additional_info['frame_tags']}")
                del tag.additional_info["frame_tags"]
        return tag

    def _fix_frame_indices(self, tag: Tag, offset: float, fps: float) -> Tag:
        """Fix frame indices (called from actor thread)"""
        if "frame_tags" not in tag.additional_info:
            return tag

        frame_tags = tag.additional_info["frame_tags"]
        frame_offset = round(offset * fps)
        residual = (offset * fps) - frame_offset
        if not math.isclose(residual, 0.0, abs_tol=1e-6):
            logger.warning(f"Non-integer frame offset detected\noffset: {offset}, fps: {fps}, frame_offset: {frame_offset}, residual: {residual}")

        adjusted = {}
        for frame_idx, label in frame_tags.items():
            try:
                frame_idx = int(frame_idx)
            except ValueError:
                logger.error(f"Invalid frame index: {tag}")
                continue
            adjusted[frame_idx + frame_offset] = label

        tag.additional_info["frame_tags"] = adjusted
        return tag
    
    def _assign_default_streams(self, args: TagArgs) -> TagArgs:
        """Assign default streams (called from actor thread)"""
        for feature, config in args.features.items():
            if config.stream is None:
                model_config = self.cregistry.get_model_config(feature)
                model_type = model_config.type
                if model_type in ("video", "frame"):
                    stream = "video"
                else:
                    stream = "audio"
                config.stream = stream
        return args
