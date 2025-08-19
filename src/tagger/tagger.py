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
                lock=threading.Lock()
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

        job.status.failed += failed

        # 2. tag
        container, reqresources = get_container(job.args.feature, media_files)
        taggingdone = threading.Event()
        uuid = self.manager.start(container, reqresources, taggingdone)
        job.taghandle = uuid
        job.status.status = "Tagging content"
        taggingdone.wait()

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

        return res

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
        active_jobs = self.job_store.active_jobs
        with self.store_lock:
            job_keys = active_jobs[qhit].keys()
            feature_job_keys = [job_key for job_key in job_keys if job_key[1] == feature]
            
            if len(feature_job_keys) == 0:
                raise MissingResourceError(
                    f"No job running for {feature} on {qhit}"
                )
            
            jobs = [active_jobs[qhit][job_key] for job_key in feature_job_keys]
            for job in jobs:
                job.stop_event.set()
        
    def cleanup(self) -> None:
        self.shutdown_signal.set()
        self.manager.shutdown()

    def _video_tag(
            self, 
            job: Job, 
            start_time: int | None, 
            end_time: int | None
        ) -> None:
        media_files, failed = self._download_content(job, start_time=start_time, end_time=end_time)
        job.media_files = media_files
        job.failed = failed
        self._submit_tag_job(job)

    def _image_tag(
            self,
            job: Job,
            assets: list[str] | None
        ) -> None:
        images, failed = self._download_content(job, assets=assets)
        deduped = list(set(images))
        if len(deduped) > 0:
            logger.warning(f"Found {len(images) - len(deduped)} duplicate images.")
        job.media_files = deduped
        job.failed = failed
        self._submit_tag_job(job)

    def _download_content(self, job: Job, **kwargs) -> tuple[list[str], list[str]]:
        media_files, failed = [], []
        stream = job.run_config.stream
        qhit = job.q.qhit

        if stream == "image":
            save_path = os.path.join(config["storage"]["images"], qhit, stream)
        else:
            save_path = os.path.join(config["storage"]["parts"], qhit, stream)

        try:
            # TODO: if waiting for lock, and stop_event is set, it will keep waiting and stop only after the lock is acquired.
            with self.download_lock[(qhit, stream)]:
                job.status = "Fetching content"
                with self.dl_sem:
                    # if fetching finished while waiting for lock, this will return immediately
                    if stream == "image":
                        media_files, failed = fetch_assets(job.q, save_path,  **kwargs)
                    else:
                        media_files, failed =  download_stream(job.q, stream, save_path, **kwargs, exit_event=job.stop_event)
            logger.debug(f"got list of media files {media_files}")
        if self._check_exit(job):
            return [], []
        return media_files, failed

    def _submit_tag_job(self, job: Job) -> None:
        if self._check_exit(job):
            return
        media_files = job.media_files
        if len(media_files) == 0:
            with self.store_lock:
                if self._is_job_stopped(job):
                    return
                self._set_stop_status(job, "Failed", f"No media files found for {job.q.qhit}")
            return
        total_media_files = len(media_files)
        if not job.replace:
            media_files = self._filter_tagged_media_files(media_files, job.q, job.run_config.stream, job.feature)
        logger.debug(f"Tag status for {job.q.qhit}: {job.feature} on {job.run_config.stream}")
        logger.debug(f"Total media files: {total_media_files}, Media files to tag: {len(media_files)}, Media files already tagged: {total_media_files - len(media_files)}")
        if len(media_files) == 0:
            with self.store_lock:
                if self._is_job_stopped(job):
                    return
                self._set_stop_status(job, "Completed", f"Tagging already complete for {job.feature} on {job.q.qhit}")
            return
        if len(job.allowed_gpus) > 0:
            # model will run on a gpu
            with self.store_lock:
                # TODO: Probably don't need lock
                job.media_files = media_files
                job.status = "Waiting to be assigned GPU"
            self.gpu_queue.put(job)
        else:
            with self.store_lock:
                job.media_files = media_files
                job.status = "Waiting for CPU resource"
            self.cpu_queue.put(job)
        self._watch_job(job)

    def _check_exit(self, job: Job) -> bool:
        """
        Returns True if the job has received a stop signal, False otherwise. 
        Also, sets the status of the job to "Stopped" if it has been stopped.

        Args:
            job (Job): Job to check

        Returns:
            bool: True if the job has been stopped, False
        """
        if job.stop_event.is_set():
            with self.store_lock:
                if self._is_job_stopped(job):
                    return True
                self._set_stop_status(job, "Stopped")
            return True
        return False
    
    def _get_job_status(self, job: Job) -> JobStatus:
        if job.time_ended is None:
            time_running = time.time() - job.time_started
        else:
            time_running = job.time_ended - job.time_started
        
        if job.status == "Tagging content":
            with self.filesystem_lock:
                # read how many files have been tagged
                tag_dir = os.path.join(config["storage"]["tmp"], job.feature, job.tag_job_id)
                if not os.path.exists(tag_dir):
                    tagged_files = 0
                else:
                    tagged_files = len(os.listdir(tag_dir))
            if job.run_config.stream != "image" and config["services"][job.feature].get("frame_level", False):
                # for frame level tagging on video we have two tag files per part, so we need to divide by two.
                tagged_files = tagged_files // 2
            to_tag = len(job.media_files)
            progress = f"{tagged_files}/{to_tag}"
        elif job.status == "Completed":
            tagged_files = len(job.media_files)
            progress = f"{tagged_files}/{tagged_files}"
        else:
            progress = ""

        return JobStatus(status=job.status, time_running=time_running, tagging_progress=progress, tag_job_id=job.tag_job_id, error=job.error, failed=job.failed)

    def _is_job_stopped(self, job: Job) -> bool:
        # not thread safe, call with lock
        return job.status == "Stopped" or job.status == "Failed"
    
    def _set_stop_status(
            self, 
            job: Job, 
            status: str, 
            error: str | None=None
        ) -> None:
        # Not thread safe
        # Should be called with lock
        qhit, stream, feature = job.q.qhit, job.run_config.stream, job.feature
        job = self.job_store.active_jobs[qhit][(stream, feature)]
        job.status = status
        job.error = error
        job.time_ended = time.time()
        self.job_store.inactive_jobs[qhit][(stream, feature)] = job
        del self.job_store.active_jobs[qhit][(stream, feature)]

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

    def _job_starter(
            self, 
            job_type: Literal["cpu"] | Literal["gpu"], 
            tag_queue: Queue
        ) -> None:
        """
        Args:
            wait_for_gpu (bool): if True, then the job starter will wait for a GPU to be available before starting the job
            tag_queue (Queue): queue for jobs waiting to be submitted
        """
        while True:
            # blocks until a job is available
            job = tag_queue.get()
            if self._check_exit(job):
                continue

            def wait_for_resource():
                # wait for a GPU to be available
                if job_type == "gpu":
                    return self.manager.await_gpu(timeout=config["devices"]["wait_for_gpu_sleep"])
                elif job_type == "cpu":
                    return self.manager.await_cpu(timeout=config["devices"]["wait_for_gpu_sleep"])
                else:
                    raise ValueError(f"Unknown job type: {job_type}")

            stopped = False
            while not wait_for_resource():
                # check if the job has been stopped, if not then we go back to waiting for a GPU
                if self._check_exit(job):
                    stopped = True
                    break
            if stopped:
                continue

            try:
                job_id = self.manager.run(job.feature, job.run_config.model, job.media_files, job.allowed_gpus, job.allowed_cpus, logs_subpath=job.q.qhit)
                with self.store_lock:
                    if self._is_job_stopped(job):
                        # if the job has been stopped while the container was starting
                        self.manager.stop(job.tag_job_id)
                        continue
                    job.tag_job_id = job_id
                    job.status = "Tagging content"
                logger.success(f"Started running {job.feature} on {job.q.qhit}")
            except NoResourceAvailable as e:
                # This error can happen if the model can only run on a subset of GPUs. 
                job.error = "Tried to assign GPU or CPU slot, but no suitable one was found. The job was placed back on the queue."
                # if no CPU slot or GPU is available, then we put the job back on the queue
                now = time.time()
                if (now - job.reput_time) < config["devices"]["wait_for_gpu_sleep"]:
                    # if the job was put back on the queue too recently, then we wait for a bit before putting it back
                    time.sleep(config["devices"]["wait_for_gpu_sleep"] - (now - job.reput_time))
                job.reput_time = now
                tag_queue.put(job)
                continue
            except Exception as e:
                # handle the unexpected
                logger.error(f"{traceback.format_exc()}")
                with self.store_lock:
                    self._set_stop_status(job, "Failed", str(e))

    def watch_job(self, job: Job) -> None:
        while True:
            time.sleep(config["watcher"]["sleep"])

            if job.stop_event.is_set():
                self.manager.stop(job.tag_job_id)
                with self.store_lock:
                    self._set_stop_status(job, "Stopped")
                break

            status = self.manager.status(job.tag_job_id)

            try:
                with self.filesystem_lock:
                    self._copy_new_files(job, status.tags)
            except Exception as e:
                logger.exception(f"Error copying files for job {job.q.qhit}/{job.feature}: {e}")
                with self.store_lock:
                    self._set_stop_status(job, "Failed", str(e))
            
            if status.status == "Running":
                continue

            if status.status == "Completed":
                logger.success(f"Finished running {job.feature} on {job.q.qhit}")
                try:
                    with self.filesystem_lock:
                        # move outputted tags to their correct place
                        # lock in case of race condition with status or finalize calls
                        self._move_files(job, status.tags)
                except Exception as e:
                    logger.exception(f"Error moving files for job {job.q.qhit}/{job.feature}: {e}")

            job.status = status.status
            if status.status == "Failed":
                job.error = "An error occurred while running model container"
            job.time_ended = time.time()
            with self.store_lock:
                # move job to inactive_jobs
                self.job_store.inactive_jobs[qhit][(stream, feature)] = job
                del active_jobs[qhit][(stream, feature)]

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