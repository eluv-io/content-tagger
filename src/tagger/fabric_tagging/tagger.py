
import threading
from collections import defaultdict
import time
import math
from copy import deepcopy

from loguru import logger

from src.tags.tagstore.types import Tag
from src.fetch.types import DownloadRequest
from src.tag_containers.containers import ContainerRegistry
from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tagger.fabric_tagging.types import *
from src.common.content import Content
from src.common.errors import MissingResourceError
from src.fetch.fetch_video import Fetcher
from src.tags.tagstore.tagstore import FilesystemTagStore
    
class FabricTagger:

    """
    Handles the flow of downloading data from fabric, tagging, and uploading
    """

    def __init__(
            self, 
            system_tagger: SystemTagger,
            cregistry: ContainerRegistry,
            tagstore: FilesystemTagStore,
            fetcher: Fetcher
        ):

        self.system_tagger = system_tagger
        self.cregistry = cregistry
        self.tagstore = tagstore
        self.fetcher = fetcher

        self.storelock = threading.RLock()
        self.jobstore = JobStore()
        
        self.shutdown_signal = threading.Event()

        threading.Thread(target=self._uploader, daemon=True).start()

    def tag(self, q: Content, args: TagArgs) -> dict:
        self._validate_args(args)
        args = self._assign_default_streams(args)
        logger.debug(args)
        status = {}
        for feature in args.features:
            tsjob = self.tagstore.start_job(
                qhit=q.qhit,
                track=feature,
                stream=args.features[feature].stream,
                author="tagger"
            )

            job = TagJob(
                state=JobState.starting(),
                args=JobArgs(
                    q=q,
                    feature=feature,
                    replace=args.replace,
                    runconfig=args.features[feature],
                    scope=args.scope
                ),
                stopevent=threading.Event(),
                upload_job=tsjob.id
            )

            err = self._start_job(job)
            if err:
                status[feature] = err
            else:
                status[feature] = "Job started successfully"

        return status

    def status(self, qhit: str) -> dict[str, dict[str, JobStatus]]:
        """
        Args:
            qhit (str): content object, hash, or write token that files belong to
        Returns:
            dict[str, dict[str, JobStatus]]: a dictionary mapping stream -> feature -> JobStatus
        """
        with self.storelock:
            active_jobs = self.jobstore.active_jobs
            inactive_jobs = self.jobstore.inactive_jobs

            jobids = {jobid for jobid in active_jobs.keys() if jobid.qhit == qhit}
            jobids |= {jobid for jobid in inactive_jobs.keys() if jobid.qhit == qhit}

            if len(jobids) == 0:
                raise MissingResourceError(
                    f"No jobs started for {qhit}"
                )

            res = defaultdict(dict)
            for jid in jobids:
                job = active_jobs.get(jid) or inactive_jobs[jid]
                stream, feature = job.args.runconfig.stream, job.args.feature
                res[stream][feature] = self._summarize_status(job.state)

        return dict(res)

    def stop(self, qhit: str, feature: str) -> None:
        with self.storelock:
            active_jobs = self.jobstore.active_jobs
            jobids = {jobid for jobid in active_jobs.keys() if jobid.qhit == qhit and jobid.feature == feature}

            if len(jobids) == 0:
                old_jobs = [jobid for jobid in self.jobstore.inactive_jobs.keys() if jobid.qhit == qhit and jobid.feature == feature]
                if old_jobs:
                    raise MissingResourceError(f"Job for {feature} on {qhit} is already complete")
                raise MissingResourceError(
                    f"No job running for {feature} on {qhit}"
                )

            jobs = [active_jobs[jid] for jid in jobids]
            for job in jobs:
                self._set_stop_state(job.get_id(), "Stopped", "", None)

        for job in jobs:
            if not job.state.taghandle:
                continue

            try:
                self.system_tagger.stop(job.state.taghandle)
            except Exception as e:
                logger.error(f"Error stopping job {job.get_id()}: {e}")

    def cleanup(self) -> None:
        logger.info("Shutting down fabric tagger...")
        with self.storelock:
            active_jobs = list(self.jobstore.active_jobs.keys())
            for job in active_jobs:
                self.stop(job.qhit, job.feature)
        self.shutdown_signal.set()
        self.system_tagger.shutdown()

    def _validate_args(self, args: TagArgs) -> None:
        """
        Raises error if args are bad
        """

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

    def _start_job(self, job: TagJob) -> str:
        # Check if job is running else start it up in new thread

        with self.storelock:
            jobid = job.get_id()

            if jobid in self.jobstore.active_jobs:
                return f"Job {(jobid.qhit, jobid.feature, jobid.stream)} is already running"

            self.jobstore.active_jobs[jobid] = job

            threading.Thread(target=self._run_job, args=(job, )).start()

        return ""

    def _run_job(self, job: TagJob) -> None:

        logger.debug(f"Tag args: {job.args}")

        # 1. download
        with self.storelock:
            if job.stopevent.is_set():
                return
            job.state.status.status = "Fetching content"

        jobid = job.get_id()

        try:
            dl_res = self.fetcher.download(
                job.args.q, 
                DownloadRequest(
                    stream_name=job.args.runconfig.stream,
                    scope=job.args.scope,
                    preserve_track=job.args.feature if not job.args.replace else "",
                ),
                exit_event=job.stopevent
            )
        except Exception as e:
            self._set_stop_state(jobid, "Failed", "Failed to download content", e)
            return

        if len(dl_res.successful_sources) == 0:
            self._set_stop_state(jobid, "Completed", "Nothing left to tag", None)
            return

        # 2. tag
        with self.storelock:
            if job.stopevent.is_set():
                return
            try:
                taggingdone = self._start_tagging_phase(job, dl_res)
            except Exception as e:
                self._set_stop_state(jobid, "Failed", "", e)
                return

        taggingdone.wait()

        # check status
        state = self.system_tagger.status(job.state.taghandle)
        if state.status not in ("Completed", "Stopped", "Failed"):
            self._set_stop_state(jobid, "Failed", "", RuntimeError(f"System tagger gave unexpected status: {state.status}"))

        if state.status in ("Stopped", "Failed"):
            self._set_stop_state(jobid, state.status, "", state.error)
            return

        try:
            self._upload_tags(job)
        except Exception as e:
            self._set_stop_state(jobid, "Failed", "", e)
            return

        self._set_stop_state(jobid, "Completed", "All tags uploaded successfully", None)

    def _start_tagging_phase(self, job: TagJob, dl_res: DownloadResult) -> threading.Event:
        with self.storelock:
            # set tagging state
            job.state.media = dl_res
            job.state.status.failed += dl_res.failed
            media_files = [s.filepath for s in dl_res.successful_sources]
            container = self.cregistry.get(job.args.feature, media_files, job.args.runconfig.model)
            reqresources = self.cregistry.get_model_config(job.args.feature).resources
            taggingdone = threading.Event()
            # TODO: holding storelock while container is starting.
            uid = self.system_tagger.start(container, reqresources, taggingdone)
            job.state.container = container
            job.state.taghandle = uid
            job.state.status.status = "Tagging content"
            return taggingdone

    def _set_stop_state(
            self, 
            jobid: JobID, 
            status: JobStateDescription,
            message: str,
            error: Exception | None,
        ) -> None:
        if error:
            logger.exception(error)
        with self.storelock:
            if jobid in self.jobstore.active_jobs:
                job = self.jobstore.active_jobs[jobid]
                job.stopevent.set()

                job.state.status.status = status
                job.state.status.time_ended = time.time()

                job.state.message = message

                self.jobstore.inactive_jobs[jobid] = job
                del self.jobstore.active_jobs[jobid]
            else:
                logger.warning(f"Tried to stop an inactive job: {jobid}")

    def _summarize_status(self, state: JobState) -> dict:
        with self.storelock:
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

    def _uploader(self) -> None:
        while not self.shutdown_signal.is_set():
            if self.storelock.acquire(timeout=1):
                try:
                    self._upload_all()
                except Exception as e:
                    logger.exception(f"Unexpected error in uploader: {e}")
                finally:
                    self.storelock.release()
            time.sleep(0.2)

    def _upload_all(self) -> None:
        with self.storelock:
            jobs = list(self.jobstore.active_jobs.values())
            jobs += list(self.jobstore.inactive_jobs.values())
        for job in jobs:
            self._upload_tags(job)

    def _upload_tags(
            self, 
            job: TagJob,
        ) -> None:
        with self.storelock:
            if job.state.container is None:
                return
            assert job.state.media is not None
            media_to_source = {s.filepath: s for s in job.state.media.successful_sources}
            outputs = job.state.container.tags()
            new_outputs = [out for out in outputs if media_to_source[out.source_media].name not in job.state.uploaded_sources]

            stream_meta = job.state.media.stream_meta
            tags2upload = []
            for out in new_outputs:
                original_src = media_to_source[out.source_media]
                for tag in out.tags:
                    # add missing metadata
                    tag.source = original_src.name
                    tag.jobid = job.upload_job
                    tags2upload.append(self._fix_tag_offsets(tag, original_src.offset, stream_meta.fps if stream_meta else None))
                    
            job.state.uploaded_sources.extend(media_to_source[out.source_media].name for out in new_outputs)
            self.tagstore.upload_tags(tags2upload, job.upload_job)

    def _fix_tag_offsets(
            self, 
            tag: Tag, 
            offset: float,
            fps: float | None,
        ) -> Tag:
        """
        Adjust tag offsets and set the source/jobid fields appropriately so they can be uploaded to tagstore
        """
        if tag.start_time is not None:
            tag.start_time += int(offset * 1000) # convert to ms
        if tag.end_time is not None:
            tag.end_time += int(offset * 1000)
        if "frame_tags" in tag.additional_info:
            assert fps is not None
            tag = self._fix_frame_indices(tag, offset, fps)
        return tag

    def _fix_frame_indices(self, tag: Tag, offset: float, fps: float) -> Tag:
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