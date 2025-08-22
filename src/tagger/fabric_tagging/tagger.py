
import threading
from collections import defaultdict
import time
import uuid

from loguru import logger

from src.common.schema import Tag
from src.fetch.types import VodDownloadRequest
from src.tag_containers.containers import ContainerRegistry
from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tagger.fabric_tagging.types import Job, JobArgs, JobStatus, JobState, JobStore, JobID
from src.common.content import Content
from src.api.tagging.format import TagArgs, ImageTagArgs
from src.common.errors import MissingResourceError
from src.fetch.fetch_video import Fetcher, VodDownloadRequest
from src.fetch.types import Source, StreamMetadata
from src.tags.tagstore import FilesystemTagStore, Job as TagJob

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

        self.dblock = threading.Lock()

        self.storelock = threading.Lock()
        self.jobstore = JobStore()

        # map qid -> token
        self.tokens = {}
        
        self.shutdown_signal = threading.Event()

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

    def tag(self, q: Content, args: TagArgs | ImageTagArgs) -> dict:
        self._validate_args(args)
        # TODO: handle image
        if not isinstance(args, TagArgs):
            raise NotImplementedError("Image tagging is not implemented yet")
        status = {}
        for feature in args.features:
            job = Job(
                args=JobArgs(
                    q=q,
                    feature=feature,
                    replace=args.replace,
                    runconfig=args.features[feature],
                    start_time=args.start_time,
                    end_time=args.end_time,
                ),
                status=JobStatus.starting(),
                taghandle="",
                lock=threading.Lock(),
                stopevent=threading.Event(),
                media=None,
                container=None,
                uploaded_sources=[]
            )
            err = self._start_job(job)
            if err:
                status[feature] = err
            else:
                status[feature] = "Job started successfully"
        return status

    def _start_job(self, job: Job) -> str:
        with self.storelock:
            jobid = job.get_id()

            if jobid in self.jobstore.active_jobs:
                return f"Job {jobid} is already running"

            self.jobstore.active_jobs[jobid] = job

            if isinstance(job.args, ImageTagArgs):
                return "Image tagging is not implemented yet"
            else:
                threading.Thread(target=self._run_job, args=(job, )).start()

        return ""

    def _stop_job(self, jobid: JobID, status: JobState, error: Exception | None) -> None:
        logger.exception(error)
        with self.storelock:
            if jobid in self.jobstore.active_jobs:
                job = self.jobstore.active_jobs[jobid]

                job.status.status = status
                job.status.time_ended = time.time()

                self.jobstore.inactive_jobs[jobid] = job
                del self.jobstore.active_jobs[jobid]
            else:
                logger.warning(f"Tried to stop an inactive job: {jobid}")

    def _run_job(self, job: Job) -> None:
        jobid = job.get_id()

        tsjob = TagJob(
            id=str(uuid.uuid4()),
            qhit=job.args.q.qhit,
            track=job.args.feature,
            stream=job.args.runconfig.stream,
            timestamp=time.time(),
            author="tagger"
        )

        self.tagstore.start_job(tsjob)

        # 1. download
        dl_req = VodDownloadRequest(
            stream_name=job.args.runconfig.stream,
            start_time=job.args.start_time,
            end_time=job.args.end_time,
            preserve_track=job.args.feature if job.args.replace else "",
        )
        job.status.status = "Fetching content"
        try:
            dl_res = self.fetcher.download_stream(job.args.q, dl_req, exit_event=job.stopevent)
        except Exception as e:
            self._stop_job(jobid, "Failed", e)
            return

        if job.stopevent.is_set():
            return
        
        job.media = dl_res
        job.status.failed += dl_res.failed

        media_files = [s.filepath for s in dl_res.successful_sources]

        # 2. tag
        with self.storelock: # TODO: better way?
            container = self.cregistry.get(job.args.feature, media_files, job.args.runconfig.model)
            reqresources = self.cregistry.get_model_resources(job.args.feature)
            taggingdone = threading.Event()
            uid = self.manager.start(container, reqresources, taggingdone)
            job.container = container
            job.taghandle = uid
            job.status.status = "Tagging content"

        taggingdone.wait()

        if job.stopevent.is_set():
            return

        # 3. upload
        self._stop_job(jobid, "Completed", None)

    def status(self, qhit: str) -> dict[str, dict[str, JobStatus]]:
        """
        Args:
            qhit (str): content object, hash, or write token that files belong to
        Returns:
            dict[str, dict[str, JobStatus]]: a dictionary mapping stream -> feature -> JobStatus
        """

        active_jobs = self.jobstore.active_jobs
        inactive_jobs = self.jobstore.inactive_jobs

        with self.storelock:
            jobids = {jobid for jobid in active_jobs.keys() if jobid.qhit == qhit}
            jobids |= {jobid for jobid in inactive_jobs.keys() if jobid.qhit == qhit}

            if len(jobids) == 0:
                raise MissingResourceError(
                    f"No jobs started for {qhit}"
                )

            res = defaultdict(dict)
            for jid in jobids:
                job = active_jobs.get(jid, inactive_jobs[jid])
                stream, feature = job.args.runconfig.stream, job.args.feature
                res[stream][feature] = self._summarize_status(job.status)

        return dict(res)

    def _summarize_status(self, status: JobStatus) -> dict:
        end = time.time()
        if status.time_ended:
            end = status.time_ended
        return {
            "status": status.status,
            "time_running": end - status.time_started,
            "tagging_progress": status.tagging_progress,
            "failed": status.failed,
        }

    def stop(self, qhit: str, feature: str) -> None:
        active_jobs = self.jobstore.active_jobs
        with self.storelock:
            jobids = {jobid for jobid in active_jobs.keys() if jobid.qhit == qhit and jobid.feature == feature}

            if len(jobids) == 0:
                raise MissingResourceError(
                    f"No job running for {feature} on {qhit}"
                )

            jobs = [active_jobs[jid] for jid in jobids]
            for job in jobs:
                job.stopevent.set()

                if job.taghandle:
                    try:
                        self.manager.stop(job.taghandle)
                    except Exception as e:
                        logger.error(f"Error stopping job {job.get_id()}: {e}")

    def _job_watcher(self) -> None:
        while not self.shutdown_signal.is_set():
            try:
                with self.storelock:
                    self._check_jobs()
            except Exception as e:
                logger.exception(f"Unexpected error in job watcher: {e}")
            time.sleep(0.2)

    # TODO: might miss tags at the end of the job
    def _check_jobs(self) -> None:
        for jobid, job in self.jobstore.active_jobs.items():
            if not job.status.status == "Tagging content":
                continue

            self._upload_tags(job)

    def _upload_tags(
            self, 
            job: Job,
        ) -> None:
        if job.container is None:
            return
        assert job.media is not None
        media_to_source = {s.filepath: s for s in job.media.successful_sources}
        outputs = job.container.tags()
        outputs = [out for out in outputs if out.source_media not in job.uploaded_sources]

        tags2upload = []
        for out in outputs:
            original_src = media_to_source[out.source_media]
            for tag in out.tags:
                tags2upload.append(self._correct_tag(tag, original_src, job.media.stream_meta))

        job.uploaded_sources.extend(out.source_media for out in outputs)
        self.tagstore.upload_tags(tags2upload, job.taghandle)

    def _correct_tag(self, tag: Tag, source: Source, meta: StreamMetadata | None) -> Tag:
        if tag.start_time is not None:
            tag.start_time += int(source.offset * 1000) # convert to ms
        if tag.end_time is not None:
            tag.end_time += int(source.offset * 1000)
        tag.source = source.name
        if "frame_tags" in tag.additional_info:
            assert meta is not None
            assert meta.fps is not None
            tag = self._correct_frame_indices(tag, source, meta.fps)
        return tag

    def _correct_frame_indices(self, tag: Tag, source: Source, fps: float) -> Tag:
        if "frame_tags" not in tag.additional_info:
            return tag

        frame_tags = tag.additional_info["frame_tags"]
        frame_offset = int(tag.start_time / 1000 * fps) + int(source.offset * fps)

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

    def cleanup(self) -> None:
        self.shutdown_signal.set()
        self.manager.shutdown()