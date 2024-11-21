import pynvml
import atexit
import uuid
import os
import threading
import uuid
import podman
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
import time

from containers import create_container
from config import config

class NoGPUAvailable(Exception):
    pass

@dataclass
class TagJob:
    container: podman.Container
    feature: str
    logs_out: str
    device: int
    status: str
    stop_event: threading.Event
    time_started: float
    time_ended: Optional[float]
    tags: List[str]
    warnings: List[str]

class ResourceManager:
    def __init__(self):
        pynvml.nvmlInit()
        self.client = podman.PodmanClient()
        atexit.register(self.shutdown)
        self.num_devices = pynvml.nvmlDeviceGetCount()
        self.device_status = [is_gpu_in_use(i) for i in range(self.num_devices)]
        self.lock = threading.Lock()
        self.jobs = {}
        # files currently being tagged + model pair, cannot tag same file twice with same model
        self.files_tagging = set()

    def run(self, feature, files) -> uuid.UUID:
        with self.lock:
            device_idx = None
            for i, status in enumerate(self.device_status):
                if not status:
                    if is_gpu_in_use(i):
                        logger.error(f"GPU {i} is in use by another process")
                        continue
                    device_idx = i
                    self.device_status[i] = True
                    break
            if device_idx is None:
                raise NoGPUAvailable("No available GPUs")
            jobid = uuid.uuid4()
            logs_out = os.path.join(config["storage"]["log"], f"{jobid}.log")
            container = create_container(self.client, feature, files, logs_out)
            for f in files:
                if (f, feature) in self.files_tagging:
                    raise ValueError(f"File {f} is already being tagged with {feature}")
                self.files_tagging.add((f, feature))
            container.start()
            self.jobs[jobid] = TagJob(container, feature, logs_out, device_idx, "running", threading.Event(), time.time(), [], [])

        threading.Thread(target=self._watch_job, args=(jobid, feature, files)).start()
        return jobid
    
    def _watch_job(self, jobid: uuid.UUID, feature: str, files: List[str]) -> None:
        job = self.jobs[jobid]
        with open(job.logs_out, "w") as fout:
            for log in job.container.logs(stream=True, stderr=True, stdout=True):
                if job.stop_event.is_set():
                    break
                fout.write(log.decode("utf-8"))
                time.sleep(1)
        if job.stop_event.is_set():
            job.container.stop()
            job.container.remove()
            with self.lock:
                self.device_status[job.device] = False
                self.jobs[jobid].status = "stopped"
            return
        job.container.wait()
        tags = []
        with self.lock:
            for f in files:
                video_tags = os.path.join(config["storage"]["tmp"], feature, f"{os.path.basename(f).split('.')[0]}_tags.json")
                frame_tags = os.path.join(config["storage"]["tmp"], feature, f"{os.path.basename(f).split('.')[0]}_frametags.json")
                if os.path.exists(video_tags):
                    tags.append(video_tags)
                else:
                    job.warnings.append(f"File {f} failed to tag")
                if os.path.exists(frame_tags) and config["services"][feature].get("frame_level", False):
                    tags.append(frame_tags)
                elif config["services"][feature].get("frame_level", False):
                    job.warnings.append(f"File {f} failed to tag frames")
                self.files_tagging.remove((f, feature))
        
            self.jobs[jobid].tags = tags
            self.jobs[jobid].status = "completed"
            self.device_status[job.device] = False

    def stop(self, jobid: uuid.UUID) -> None:
        job = self.jobs[jobid]
        job.stop_event.set()
        while job.status != "stopped":
            time.sleep(1)
        logger.info(f"Job {jobid} stopped")

    def status(self, jobid: uuid.UUID) -> dict:
        job = self.jobs[jobid]
        if job.status != "running":
            assert job.time_ended is not None
            return {"status": job.status, "took": f"{job.time_ended - job.time_started} seconds"}
        else:
            return {"status": self.jobs[jobid].status, "time running": f"{time.time() - self.jobs[jobid].time_started} seconds"}
        
    def shutdown(self):
        for jobid in self.jobs:
            self.stop(jobid)
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