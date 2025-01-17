import pynvml
import atexit
import uuid
import os
import threading
import podman
from podman.domain.containers import Container
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict
from loguru import logger
import time

from src.containers import create_container
from config import config

class NoGPUAvailable(Exception):
    pass

@dataclass
class TagJob:
    # podman container instance
    container: Container
    # model name
    feature: str
    # path to logs
    logs_out: str
    # device index
    device: int
    # status of the job
    status: Literal["Running", "Completed", "Stopped", "Failed"]
    stop_event: threading.Event
    time_started: float
    time_ended: Optional[float]
    # list of media files that are being tagged
    media_files: List[str]
    # list of output tag files
    tags: List[str]
    warnings: List[str]

class ResourceManager:
    def __init__(self):
        pynvml.nvmlInit()
        self.client = podman.PodmanClient()
        atexit.register(self.shutdown)
        self.num_devices = pynvml.nvmlDeviceGetCount()

        # ensures thread safety of device_status, jobs, and files_tagging
        self.lock = threading.Lock()

        self.device_status = [False for i in range(self.num_devices)]
        self.jobs: Dict[str, TagJob] = {}
        # (media file, model) pairs, cannot have two jobs going on the same file with the same model
        self.files_tagging = set()

    # Args:
    #     feature (str): The feature to tag the files with.
    #     run_config (dict): The configuration to run the model with. This is model-specific. Check the model's documentation.
    #     files (List[str]): The list of files to tag.
    # Returns:
    #     str: The job ID.
    # NOTE: this function creates a new thread to watch the job
    def run(self, feature: str, run_config: dict, files: List[str]) -> str:
        with self.lock:
            device_idx = None
            container = None
            files_added = []
            try:
                for i, status in enumerate(self.device_status):
                    if not status:
                        if not config["devices"]["allow_in_use_gpus"] and is_gpu_in_use(i):
                            logger.error(f"GPU {i} is in use by a non-tagger process")
                            continue
                        device_idx = i
                        self.device_status[i] = True
                        break
                if device_idx is None:
                    raise NoGPUAvailable("No available GPUs") 
                jobid = str(uuid.uuid4())
                if not os.path.exists(os.path.join(config["storage"]["logs"], feature)):
                    os.makedirs(os.path.join(config["storage"]["logs"], feature))
                logs_out = os.path.join(config["storage"]["logs"], feature, f"{jobid}.log")
                for f in files:
                    if (f, feature) in self.files_tagging:
                        raise ValueError(f"File {f} is already being tagged with {feature}")
                    self.files_tagging.add((f, feature))
                    files_added.append((f, feature))
                container = create_container(self.client, feature, files, run_config, device_idx, logs_out)
                container.start()
                self.jobs[jobid] = TagJob(container, feature, logs_out, device_idx, "Running", threading.Event(), time.time(), None, files, [], [])
            except Exception as e:
                # cleanup resources if job fails to start
                if container:
                    self._stop_container(container)
                for f, feature in files_added:
                    self.files_tagging.remove((f, feature))
                if device_idx is not None:
                    self.device_status[device_idx] = False
                raise e
        threading.Thread(target=self._watch_job, args=(jobid, )).start()
        return jobid
    
    def _is_container_active(self, status: str) -> bool:
        return status == "running" or status == "created"
    
    # main function for watching and finalizing the job which is running in a container
    def _watch_job(self, jobid: str) -> None:
        # because we aren't deleting keys, we don't need to lock
        job = self.jobs[jobid]
        logger.info(f"Watching job {jobid}")
        with open(job.logs_out, "w") as fout:
            try:
                ts = 0
                while not job.stop_event.is_set() and self._is_container_active(job.container.status):
                    # refresh status
                    job.container.reload()

                    # redirect logs
                    logs = job.container.logs(stream=False, stderr=True, stdout=True, since=ts)
                    ts = int(time.time())
                    for log in logs:
                        fout.write(log.decode("utf-8"))
                    
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error while watching job {jobid}: {e}")
                with self.lock:
                    self._cleanup_job(jobid, "Failed")
                return
            finally:
                # capture remaining logs
                logs = job.container.logs(stream=False, stderr=True, stdout=True, since=ts)
                for log in logs:
                    fout.write(log.decode("utf-8"))

        if job.stop_event.is_set():
            with self.lock:
                self._cleanup_job(jobid, "Stopped")
            return
        
        # TODO: Usually even if the container fails, exit code is 0. Need to fix. 
        exit_code = job.container.attrs["State"]["ExitCode"]
        if exit_code != 0:
            logger.error(f"Job {jobid} failed to complete")
            with self.lock:
                self._cleanup_job(jobid, "Failed")
            return
        logger.info(f"Job {jobid} completed")

        tags = []
        for f in job.media_files:
            video_tags = os.path.join(config["storage"]["tmp"], job.feature, f"{os.path.basename(f).split('.')[0]}_tags.json")
            frame_tags = os.path.join(config["storage"]["tmp"], job.feature, f"{os.path.basename(f).split('.')[0]}_frametags.json")
            image_tags = os.path.join(config["storage"]["tmp"], job.feature, f"{os.path.basename(f).split('.')[0]}_imagetags.json")
            if os.path.exists(video_tags):
                tags.append(video_tags)
            if config["services"][job.feature].get("frame_level", False):
                if os.path.exists(frame_tags):
                    tags.append(frame_tags)
                if os.path.exists(image_tags):
                    tags.append(image_tags)
            # TODO: should add warning if nothing is generated
            
        with self.lock:
            self.jobs[jobid].tags = tags
            self._cleanup_job(jobid, "Completed")

    def stop(self, jobid: str) -> None:
        job = self.jobs[jobid]
        if job.status != "Running":
            return
        job.stop_event.set()
        while job.status == "Running":
            time.sleep(1)
        logger.info(f"Job {jobid} stopped")

    # cleanup job resources
    # NOTE: NOT THREAD SAFE, must be called with lock
    def _cleanup_job(self, jobid: str, status: str) -> None:
        job = self.jobs[jobid]
        job.time_ended = time.time()
        job.status = status
        self._stop_container(job.container)
        for f in job.media_files:
            self.files_tagging.remove((f, job.feature))
        self.device_status[job.device] = False

    def _stop_container(self, container: Container) -> None:
        logger.info(f"Stopping container: status={container.status}")
        if container.status == "running":
            # podman client will kill if it doesn't stop within the timeout limit
            container.stop(timeout=2)
        container.reload()
        if container.status == "running":
            logger.error(f"Container status is still \"running\" after stop. Please check the container and stop it manually.")

    @dataclass
    class TagJobStatus:
        status: str # from TagJob.status
        tags: List[str]
        time_elapsed: Optional[float]

    def status(self, jobid: str) -> TagJobStatus:
        with self.lock:
            job = self.jobs[jobid]
            if job.status != "Running":
                return ResourceManager.TagJobStatus(job.status, tags=job.tags, time_elapsed=(job.time_ended - job.time_started))
            else:
                return ResourceManager.TagJobStatus(job.status, tags=job.tags, time_elapsed=(time.time() - job.time_started))
        
    def shutdown(self):
        for jobid in self.jobs:
            if self.jobs[jobid].status == "Running":
                self.stop(jobid)
        self.client.close()
        pynvml.nvmlShutdown()

def is_gpu_in_use(gpu_idx: int) -> bool:
    """
    Check if a GPU is in use.
    Args:
        gpu_index (int): The index of the GPU to check.
    Returns:
        bool: True if the GPU is in use, False otherwise.
    """
    if not pynvml.is_initialized():
        raise RuntimeError("pynvml is not initialized. Call pynvml.nvmlInit() before calling this function.")
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
    return bool(compute_procs or graphics_procs)