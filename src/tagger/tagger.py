import threading
from queue import Queue
from collections import defaultdict
import time
import os
from typing import List, Optional, Literal
from dataclasses import dataclass, field
from requests import HTTPError

from loguru import logger

from src.tagger.jobs import JobsStore
from src.tagger.resource_manager import ResourceManager
from src.fabric.content import Content
from src.api.tagging.format import TagArgs
from src.containers import list_services
from src.api.errors import MissingResourceError, BadRequestError
from src.tagger.jobs import Job
from src.fabric.video import download_stream, StreamNotFoundError
from src.fabric.assets import fetch_assets, AssetsNotFoundException

from config import config

@dataclass
class JobStatus():
    # JobStatus represents the status of a job returned by the /status endpoint
    status: Literal["Starting", 
                    "Fetching content",
                    "Waiting to be assigned GPU", 
                    "Tagging content",
                    "Completed", 
                    "Failed", 
                    "Stopped"]
    # time running (in seconds)
    time_running: float
    tagging_progress: str
    tag_job_id: Optional[str]=None
    error: Optional[str]=None
    failed: List[str]=field(default_factory=list)

class Tagger():
    def __init__(
            self, 
            job_store: JobsStore,
            manager: ResourceManager,
        ):
        self.job_store = job_store
        self.manager = manager

        self.cpu_queue = Queue()
        self.gpu_queue = Queue()

        self.filesystem_lock = threading.Lock()

        self.store_lock = threading.Lock()
        self.job_store = job_store

        # make sure no two streams are being downloaded at the same time. 
        # maps (qhit, stream) -> lock
        self.download_lock = defaultdict(threading.Lock)

        # controls the number of concurrent downloads
        self.dl_sem = threading.Semaphore(config["fabric"]["max_downloads"])
        
        self.shutdown_signal = threading.Event()

    def tag(self, qhit: str, q: Content, args: TagArgs) -> None:
        services = list_services()

        invalid_features = [feature for feature in args.features if feature not in services]

        if len(invalid_features) > 0:
            raise MissingResourceError(
                f"Invalid features: {', '.join(invalid_features)}. Available features: {', '.join(services)}"
            )
        
        with self.store_lock:
            if self.shutdown_signal.is_set():
                raise RuntimeError("Tagger is shutting down, cannot start new jobs")
            for feature, run_config in args.features.items():
                if run_config.stream is None:
                    # if stream name is not provided, we pick stream based on whether the model is audio/video based
                    run_config.stream = config["services"][feature]["type"]
                if (run_config.stream, feature) in self.job_store.active_jobs[q.qhit]:
                    raise BadRequestError(
                        f"{feature} tagging is already in progress for {q.qhit} on {run_config.stream}"
                    )
            for feature, run_config in args.features.items():
                # get the subset of GPUs that the model can run on, default to all of them
                allowed_gpus = config["services"][feature].get("allowed_gpus", list(range(self.manager.num_devices)))
                allowed_cpus = config["services"][feature].get("cpu_slots", [])
                logger.debug(f"Starting {feature} on {q.qid}: {run_config.stream}")
                job = Job(
                    q=q, 
                    run_config=run_config, 
                    feature=feature, 
                    media_files=[], 
                    failed=[], 
                    replace=args.replace, 
                    status="Starting", 
                    stop_event=threading.Event(), 
                    time_started=time.time(), 
                    allowed_gpus=allowed_gpus, 
                    allowed_cpus=allowed_cpus
                )
                self.job_store.active_jobs[q.qhit][(run_config.stream, feature)] = job
                threading.Thread(target=self._video_tag, args=(job, args.start_time, args.end_time)).start()

    def _video_tag(self, job: Job, start_time: int | None, end_time: int | None) -> None:
        media_files, failed = self._download_content(job, q, start_time=start_time, end_time=end_time)
        job.media_files = media_files
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
                        media_files, failed = fetch_assets(q, save_path,  **kwargs)
                    else:
                        media_files, failed =  download_stream(q, stream, save_path, **kwargs, exit_event=job.stop_event)
            logger.debug(f"got list of media files {media_files}")
        except (StreamNotFoundError, AssetsNotFoundException):
            with self.store_lock:
                self._set_stop_status(job, "Failed", f"Content for stream {stream} was not found for {qhit}")
        except HTTPError as e:
            with self.store_lock:
                self._set_stop_status(job, "Failed", f"Failed to fetch stream {stream} for {qhit}: {str(e)}. Make sure authorization token hasn't expired.")
        except Exception as e:
            with self.store_lock:
                self._set_stop_status(job, "Failed", f"Unknown error occurred while fetching stream {stream} for {qhit}: {str(e)}")
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
            media_files = self._filter_tagged_files(media_files, job.q, job.run_config.stream, job.feature)
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
            with filesystem_lock:
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
    
    def _set_stop_status(self, job: Job, status: str, error: str | None=None) -> None:
        # Not thread safe
        # Should be called with lock
        qhit, stream, feature = job.q.qhit, job.run_config.stream, job.feature
        job = self.job_store.active_jobs[qhit][(stream, feature)]
        job.status = status
        job.error = error
        job.time_ended = time.time()
        self.job_store.inactive_jobs[qhit][(stream, feature)] = job
        del self.job_store.active_jobs[qhit][(stream, feature)]

    def _filter_tagged_files(
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