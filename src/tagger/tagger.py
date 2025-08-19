import threading
from queue import Queue
from collections import defaultdict
import time
import os
from typing import List, Literal
from dataclasses import dataclass, asdict, field
import shutil
import traceback

from loguru import logger
from requests import HTTPError

from src.tagger.resource_manager import  NoResourceAvailable, SystemTagger
from src.fabric.content import Content
from src.api.tagging.format import TagArgs, ImageTagArgs
from src.api.errors import MissingResourceError, BadRequestError
from src.tagger.tagsdb import check_qid, load_tag_file, upload_tags
from src.fabric.video import download_stream, StreamNotFoundError
from src.fabric.assets import fetch_assets, AssetsNotFoundException

from config import config

JobState = Literal[
    "Starting",
    "Fetching content",
    "Waiting for resources",
    "Tagging content",
    "Completed",
    "Failed",
    "Stopped"
]

@dataclass
class JobStatus:
    status: JobState
    time_started: float
    time_ended: float | None
    tagging_progress: str
    failed: list[str]

    @staticmethod
    def starting() -> 'JobStatus':
        return JobStatus(
            status="Starting",
            time_started=time.time(),
            time_ended=None,
            tagging_progress="0%",
            failed=[]
        )

@dataclass
class RunConfig:
    # model config, used to overwrite the model level config
    model: dict
    # stream name to run the model on, None to use the default stream. "image" is a special case which will tag image assets
    stream: str

@dataclass
class JobArgs:
    q: Content
    feature: str
    runconfig: RunConfig
    start_time: int | None
    end_time: int | None

@dataclass
class Job:
    args: JobArgs
    status: JobStatus
    taghandle: str
    lock: threading.Lock
    stopevent: threading.Event

    def get_id(self) -> 'JobID':
        return JobID(qhit=self.args.q.qhit, feature=self.args.feature, stream=self.args.runconfig.stream)

@dataclass
class JobID:
    qhit: str
    feature: str
    stream: str

    def __hash__(self):
        return hash((self.qhit, self.feature, self.stream))

@dataclass
class JobStore:
    active_jobs: dict[JobID, Job] = field(default_factory=dict)
    inactive_jobs: dict[JobID, Job] = field(default_factory=dict)

class FabricTagger:

    """
    Handles the flow of downloading data from fabric, tagging, and uploading
    """

    def __init__(
            self, 
            manager: SystemTagger
        ):
        self.manager = manager

        self.dblock = threading.Lock()

        self.storelock = threading.Lock()
        self.jobstore = JobStore()

        # maps (qhit, stream) -> lock
        self.download_lock = defaultdict(threading.Lock)

        # controls the number of concurrent downloads
        self.dl_sem = threading.Semaphore(config["fabric"]["max_downloads"])

        # map qid -> token
        self.tokens = {}
        
        self.shutdown_signal = threading.Event()

    def _validate_args(self, args: TagArgs | ImageTagArgs) -> None:
        """
        Raises error if args are bad
        """

        services = list_services()  

        invalid_features = [feature for feature in args.features if feature not in services]

        if len(invalid_features) > 0:
            raise MissingResourceError(
                f"Invalid features: {', '.join(invalid_features)}. Available features: {', '.join(services)}"
            )

        if isinstance(args, ImageTagArgs):
            for feat in args.features.keys():
                if not config["services"][feat].get("frame_level", False):
                    raise MissingResourceError(
                        f"Image tagging for {feat} is not supported"
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
                    runconfig=args.features[feature],
                    start_time=args.start_time,
                    end_time=args.end_time
                ),
                status=JobStatus.starting(),
                taghandle="",
                lock=threading.Lock(),
                stopevent=threading.Event()
            )
            err = self._start_job(job)
            if err:
                status[feature] = str(err)
            else:
                status[feature] = "Job started successfully"
        return status

    def _start_job(self, job: Job) -> Exception | None:
        with self.storelock:
            jobid = job.get_id()

            if jobid in self.jobstore.active_jobs:
                return BadRequestError(f"Job {jobid} is already running")

            self.jobstore.active_jobs[jobid] = job

            if isinstance(job.args, ImageTagArgs):
                raise NotImplementedError("Image tagging is not implemented yet")
            else:
                threading.Thread(target=self._run_job, args=(job, )).start()

        return None

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

        # 1. download
        job.status.status = "Fetching content"
        try:
            media_files, failed = self._download_content(job, start_time=job.args.start_time, end_time=job.args.end_time)
        except Exception as e:
            self._stop_job(jobid, "Failed", e)
            return

        if job.stopevent.is_set():
            return

        job.status.failed += failed

        # 2. tag
        with self.storelock: # TODO: better way?
            container, reqresources = get_container(job.args.feature, media_files)
            taggingdone = threading.Event()
            uuid = self.manager.start(container, reqresources, taggingdone)
            job.taghandle = uuid
            job.status.status = "Tagging content"

        taggingdone.wait()

        if job.stopevent.is_set():
            return

        # 3. upload
        self._upload_content(job, media_files)

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

    def cleanup(self) -> None:
        self.shutdown_signal.set()
        self.manager.shutdown()

    def _download_content(self, job: Job, **kwargs) -> tuple[list[str], list[str]]:
        media_files, failed = [], []
        stream = job.args.runconfig.stream
        qhit = job.args.q.qhit

        if stream == "image":
            save_path = os.path.join(config["storage"]["images"], qhit, stream)
        else:
            save_path = os.path.join(config["storage"]["parts"], qhit, stream)

        # TODO: if waiting for lock, and stop_event is set, it will keep waiting and stop only after the lock is acquired.
        
        with self.download_lock[(qhit, stream)]:
            with self.dl_sem:
                if stream == "image":
                    media_files, failed = fetch_assets(job.args.q, save_path,  **kwargs)
                else:
                    media_files, failed =  download_stream(job.args.q, stream, save_path, **kwargs, exit_event=job.stopevent)

        if job.stopevent.is_set():
            return [], []

        return media_files, failed

    def _filter_tagged_media_files(
            self,
            media_files: list[str], 
            q: Content,
            stream: str, 
            feature: str
        ) -> List[str]:
        """
        Args:
            media_files (List[str]): list of media files to filter
            qhit (str): content object, hash, or write token that files belong to
            stream (str): stream name
            feature (str): model name

        Returns:
            List[str]: list of media files that have not been tagged, filtered subset of media_files
        """
        try:
            if stream == "image":
                tag_files = q.list_files(path=f"image_tags/{feature}")
            else:
                tag_files = q.list_files(path=f"video_tags/{stream}/{feature}")
        except HTTPError:
            # if the folder doesn't exist, then no files have been tagged
            return media_files[:]
        tagged = set(self._source_from_tag_file(tag) for tag in tag_files)
        untagged = []
        for media_file in media_files:
            filename = os.path.basename(media_file)
            if filename not in tagged:
                untagged.append(media_file)
        return untagged


    def _filter_tagged_files(
            self,
            tagfiles: list[str], 
            q: Content,
            stream: str, 
            feature: str
        ) -> list[str]:
        """
        Args:
            media_files (List[str]): list of media files to filter
            qhit (str): content object, hash, or write token that files belong to
            stream (str): stream name
            feature (str): model name

        Returns:
            List[str]: list of media files that have not been tagged, filtered subset of media_files
        """
        try:
            if stream == "image":
                remote_files = q.list_files(path=f"image_tags/{feature}")
            else:
                remote_files = q.list_files(path=f"video_tags/{stream}/{feature}")
        except HTTPError:
            # if the folder doesn't exist, then no files have been tagged
            return tagfiles[:]
        untagged = []
        for tagfile in tagfiles:
            if not os.path.basename(tagfile) in remote_files:
                untagged.append(tagfile)
        return untagged

    def _source_from_tag_file(self, tagged_file: str) -> str:
        """
        Args:
            tagged_file (str): a tag file name, generated by tagger

        Returns:
            str: the source file name that the tag file was generated from
        """
        if tagged_file.endswith("_imagetags.json"):
            return tagged_file.split("_imagetags.json")[0]
        if tagged_file.endswith("_frametags.json"):
            return tagged_file.split("_frametags.json")[0]
        if tagged_file.endswith("_tags.json"):
            return tagged_file.split("_tags.json")[0]

        raise ValueError(f"Unknown tag file format: {tagged_file}")

    def _move_files(
            self, 
            job: Job, 
            tags: List[str]
        ) -> None:
        if len(tags) == 0:
            return
        qhit, stream, feature = job.q.qhit, job.run_config.stream, job.feature
        tags_path = os.path.join(config["storage"]["tags"], qhit, stream, feature)
        os.makedirs(tags_path, exist_ok=True)
        for tag in tags:
            shutil.move(tag, os.path.join(tags_path, os.path.basename(tag)))
        tag_dir = os.path.dirname(tags[0])
        shutil.rmtree(tag_dir, ignore_errors=True)

    def _copy_new_files(
            self, 
            job: Job, 
            tags: List[str]
        ) -> None:
        # TODO: check inodes instead of skipping last tag
        if len(tags) == 0:
            return
        qhit, stream, feature = job.q.qhit, job.run_config.stream, job.feature
        tags_path = os.path.join(config["storage"]["tags"], qhit, str(stream), feature)
        os.makedirs(tags_path, exist_ok=True)
        tagrows = []
        for tag in tags:
            if os.path.exists(os.path.join(tags_path, os.path.basename(tag))):
                continue
            shutil.copyfile(tag, os.path.join(tags_path, os.path.basename(tag)))
            # format for tags schema
            tagrows += load_tag_file(qhit, feature, os.path.join(tags_path, os.path.basename(tag)))
        
        if len(tagrows) > 0 and check_qid(qhit):
            # upload tags to the database only if tagging is against a qid
            try:
                upload_tags(qhit, tagrows)
            except Exception as e:
                logger.exception(f"Error uploading tags for {qhit}/{feature}: {e}")