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
    container: Container
    feature: str
    logs_out: str
    device: int
    status: Literal["running", "completed", "stopped"]
    stop_event: threading.Event
    time_started: float
    time_ended: Optional[float]
    media_files: List[str]
    tags: List[str]
    warnings: List[str]

class ResourceManager:
    def __init__(self):
        pynvml.nvmlInit()
        self.client = podman.PodmanClient()
        atexit.register(self.shutdown)
        self.num_devices = pynvml.nvmlDeviceGetCount()
        self.device_status = [False for i in range(self.num_devices)]
        self.lock = threading.Lock()
        self.jobs: Dict[str, TagJob] = {}
        # (media file, model) pairs, cannot have two jobs going on the same file with the same model
        self.files_tagging = set()

    # Args:
    #     feature (str): The feature to tag the files with.
    #     files (List[str]): The list of files to tag.
    # Returns:
    #     uuid.UUID: The job ID.
    # NOTE: this function creates a new thread to watch the job
    def run(self, feature: str, files: List[str]) -> str:
        with self.lock:
            device_idx = None
            for i, status in enumerate(self.device_status):
                if not status:
                    if not config["devices"]["allow_in_use_gpus"] and is_gpu_in_use(i):
                        logger.error(f"GPU {i} is in use by a non-tagger process")
                        continue
                    device_idx = i
                    self.device_status[i] = True
                    break
            if device_idx is None:
                raise NoGPUAvailable("No available GPUs") # TODO: implement queueing
            jobid = str(uuid.uuid4())
            logs_out = os.path.join(config["storage"]["logs"], f"{jobid}.log")
            container = create_container(self.client, feature, files, device_idx, logs_out)
            for f in files:
                if (f, feature) in self.files_tagging:
                    raise ValueError(f"File {f} is already being tagged with {feature}")
                self.files_tagging.add((f, feature))
            container.start()
            self.jobs[jobid] = TagJob(container, feature, logs_out, device_idx, "running", threading.Event(), time.time(), None, files, [], [])
        threading.Thread(target=self._watch_job, args=(jobid, )).start()
        return jobid
    
    # main function for watching and finalizing the job which is running in a container
    def _watch_job(self, jobid: str) -> None:
        job = self.jobs[jobid]
        logger.info(f"Watching job {jobid}")
        with open(job.logs_out, "w") as fout:
            try:
                ts = 0
                while not job.stop_event.is_set() and not job.container.status == "exited":
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
                logger.warning(f"Killing container for job {jobid}")
                job.container.kill()
                self._cleanup_job(jobid, "error")
                return
            finally:
                # capture remaining logs
                logs = job.container.logs(stream=False, stderr=True, stdout=True, since=ts)
                for log in logs:
                    fout.write(log.decode("utf-8"))

        if job.stop_event.is_set():
            with self.lock:
                self._cleanup_job(jobid, "stopped")
            return
        
        logger.info(f"Job {jobid} completed")
        exit_code = job.container.attrs["State"]["ExitCode"]
        if exit_code != 0:
            logger.error(f"Job {jobid} failed to complete")
            with self.lock:
                self._cleanup_job(jobid, "error")
            return
        tags = []
        with self.lock:
            for f in job.media_files:
                video_tags = os.path.join(config["storage"]["tmp"], job.feature, f"{os.path.basename(f).split('.')[0]}_tags.json")
                frame_tags = os.path.join(config["storage"]["tmp"], job.feature, f"{os.path.basename(f).split('.')[0]}_frametags.json")
                if os.path.exists(video_tags):
                    tags.append(video_tags)
                else:
                    job.warnings.append(f"File {f} failed to tag")
                if os.path.exists(frame_tags) and config["services"][job.feature].get("frame_level", False):
                    tags.append(frame_tags)
                elif config["services"][job.feature].get("frame_level", False):
                    job.warnings.append(f"File {f} failed to tag frames")
        
            self.jobs[jobid].tags = tags
            self._cleanup_job(jobid, "completed")

    def stop(self, jobid: str) -> None:
        job = self.jobs[jobid]
        if job.status != "running":
            return
        job.stop_event.set()
        while job.status == "running":
            time.sleep(1)
        logger.info(f"Job {jobid} stopped")

    def _cleanup_job(self, jobid: str, status: str) -> None:
        job = self.jobs[jobid]
        job.time_ended = time.time()
        job.status = status
        if job.container.status == "running":
            job.container.stop()
        for f in job.media_files:
            self.files_tagging.remove((f, job.feature))
        self.device_status[job.device] = False

    @dataclass
    class TagJobStatus:
        status: Literal["running", "completed", "stopped", "error"]
        tags: List[str]
        time_elapsed: Optional[float]

    def status(self, jobid: str) -> TagJobStatus:
        job = self.jobs[jobid]
        if job.status != "running":
            # TODO: check race condition, time_ended might be None if in process of finalizing
            return ResourceManager.TagJobStatus(job.status, tags=job.tags, time_elapsed=(job.time_ended - job.time_started))
        else:
            return ResourceManager.TagJobStatus(job.status, tags=job.tags, time_elapsed=(time.time() - job.time_started))
        
    def shutdown(self):
        for jobid in self.jobs:
            if self.jobs[jobid].status == "running":
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