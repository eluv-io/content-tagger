
import threading
from collections import defaultdict
import time
from copy import deepcopy

from loguru import logger

from src.tags.tagstore.types import Tag
from src.fetch.types import VodDownloadRequest
from src.tag_containers.containers import ContainerRegistry
from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tagger.fabric_tagging.types import TagJob, JobArgs, JobStatus, JobState, JobStore, JobID, JobStateDescription
from src.common.content import Content
from src.api.tagging.format import TagArgs, ImageTagArgs
from src.common.errors import MissingResourceError
from src.fetch.fetch_video import Fetcher, VodDownloadRequest
from src.fetch.types import Source, StreamMetadata
from src.tags.tagstore.tagstore import FilesystemTagStore
    
class FabricTagger:

    """
    Handles the flow of downloading data from fabric, tagging, and uploading
    """

    def __init__(
            self, 
            manager: SystemTagger,
            cregistry: ContainerRegistry,
            tagstore: FilesystemTagStore,
            fetcher: Fetcher
        ):

        self.manager = manager
        self.cregistry = cregistry
        self.tagstore = tagstore
        self.fetcher = fetcher

        self.storelock = threading.RLock()
        self.jobstore = JobStore()
        
        self.shutdown_signal = threading.Event()

        threading.Thread(target=self._uploader, daemon=True).start()

    def tag(self, q: Content, args: TagArgs | ImageTagArgs) -> dict:
        self._validate_args(args)
        # TODO: handle image
        if not isinstance(args, TagArgs):
            raise NotImplementedError("Image tagging is not implemented yet")
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
                    start_time=args.start_time,
                    end_time=args.end_time,
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
                raise MissingResourceError(
                    f"No job running for {feature} on {qhit}"
                )

            jobs = [active_jobs[jid] for jid in jobids]
            for job in jobs:
                self._set_stop_state(job.get_id(), "Stopped", None)

        for job in jobs:
            if job.state.taghandle is None:
                continue
            
            try:
                self.manager.stop(job.state.taghandle)
            except Exception as e:
                logger.error(f"Error stopping job {job.get_id()}: {e}")

    def cleanup(self) -> None:
        logger.info("Shutting down fabric tagger...")
        with self.storelock:
            active_jobs = list(self.jobstore.active_jobs.keys())
            for job in active_jobs:
                self.stop(job.qhit, job.feature)
        self.shutdown_signal.set()
        self.manager.shutdown()

    def _validate_args(self, args: TagArgs | ImageTagArgs) -> None:
        """
        Raises error if args are bad
        """

        services = self.cregistry.services()

        modconfigs = self.cregistry.cfg.modconfigs

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
            
            if isinstance(job.args, ImageTagArgs):
                return "Image tagging is not implemented yet"

            self.jobstore.active_jobs[jobid] = job

            threading.Thread(target=self._run_job, args=(job, )).start()

        return ""

    def _run_job(self, job: TagJob) -> None:
        # 1. download
        with self.storelock:
            if job.stopevent.is_set():
                return
            job.state.status.status = "Fetching content"

        jobid = job.get_id()

        try:
            dl_res = self.fetcher.download_stream(
                job.args.q, 
                VodDownloadRequest(
                    stream_name=job.args.runconfig.stream,
                    start_time=job.args.start_time,
                    end_time=job.args.end_time,
                    preserve_track=job.args.feature if job.args.replace else "",
                ),
                exit_event=job.stopevent
            )
        except Exception as e:
            self._set_stop_state(jobid, "Failed", e)
            return

        # 2. tag
        with self.storelock:
            if job.stopevent.is_set():
                return
            # set tagging state
            job.state.media = dl_res
            job.state.status.failed += dl_res.failed
            media_files = [s.filepath for s in dl_res.successful_sources]
            container = self.cregistry.get(job.args.feature, media_files, job.args.runconfig.model)
            reqresources = self.cregistry.get_model_config(job.args.feature).resources # TODO: should be combined with container
            taggingdone = threading.Event()
            # TODO: holding storelock while container is starting.
            uid = self.manager.start(container, reqresources, taggingdone)
            job.state.container = container
            job.state.taghandle = uid
            job.state.status.status = "Tagging content"

        taggingdone.wait()

        self._upload_tags(job)

        self._set_stop_state(jobid, "Completed", None)

    def _set_stop_state(self, jobid: JobID, status: JobStateDescription, error: Exception | None) -> None:
        if error:
            logger.exception(error)
        with self.storelock:
            if jobid in self.jobstore.active_jobs:
                job = self.jobstore.active_jobs[jobid]
                job.stopevent.set()

                job.state.status.status = status
                job.state.status.time_ended = time.time()

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
            return {
                "status": status.status,
                "time_running": end - status.time_started,
                "tagging_progress": f"{int(total_tagged / total_sources * 100)}%" if total_sources > 0 else "0%",
                "failed": status.failed,
            }

    def _uploader(self) -> None:
        while not self.shutdown_signal.is_set():
            if self.storelock.acquire(timeout=1):
                try:
                    self._check_jobs()
                except Exception as e:
                    logger.exception(f"Unexpected error in job watcher: {e}")
                finally:
                    self.storelock.release()
            time.sleep(0.2)

    # TODO: might miss tags at the end of the job
    def _check_jobs(self) -> None:
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

            tags2upload = []
            for out in new_outputs:
                original_src = media_to_source[out.source_media]
                for tag in out.tags:
                    tags2upload.append(self._fix_tag_offsets(tag, original_src, job.state.media.stream_meta))

            job.state.uploaded_sources.extend(media_to_source[out.source_media].name for out in new_outputs)
            self.tagstore.upload_tags(tags2upload, job.upload_job)

    def _fix_tag_offsets(self, tag: Tag, source: Source, meta: StreamMetadata | None) -> Tag:
        if tag.start_time is not None:
            tag.start_time += int(source.offset * 1000) # convert to ms
        if tag.end_time is not None:
            tag.end_time += int(source.offset * 1000)
        tag.source = source.name
        if "frame_tags" in tag.additional_info:
            assert meta is not None
            assert meta.fps is not None
            tag = self._fix_frame_indices(tag, source, meta.fps)
        return tag

    def _fix_frame_indices(self, tag: Tag, source: Source, fps: float) -> Tag:
        if "frame_tags" not in tag.additional_info:
            return tag

        frame_tags = tag.additional_info["frame_tags"]
        frame_offset = int(source.offset * fps)

        adjusted = {}
        for frame_idx, label in frame_tags.items():
            try:
                frame_idx = int(frame_idx)
            except ValueError:
                logger.exception(f"Invalid frame index: {tag}")
                continue
            adjusted[frame_idx + frame_offset] = label

        tag.additional_info["frame_tags"] = adjusted

        return tag