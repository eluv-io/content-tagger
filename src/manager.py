import time
import os
import uuid
import threading
import atexit
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict
import json

import pynvml
import podman
from podman.domain.containers import Container
from loguru import logger

from src.containers import create_container

from config import config

class NoResourceAvailable(Exception):
    pass

@dataclass
class TagJob:
    # podman container instance
    container: Container
    # model name
    feature: str
    # path to logs
    logs_out: str
    # device index, None if no GPU
    device: Optional[int]
    # cpu "slot" index,  None if no CPU slot
    cpu_slot: Optional[str]
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
    save_path: str
    
    def __post_init__(self):
        if self.device is not None and self.cpu_slot is not None:
            raise ValueError("a TagJob cannot have both a GPU and CPU slot")

class ResourceManager:
    def __init__(self):
        pynvml.nvmlInit()
        self.client = podman.PodmanClient()
        atexit.register(self.shutdown)
        self.num_devices = pynvml.nvmlDeviceGetCount()

        # ensures thread safety of device_status, jobs, and files_tagging
        self.lock = threading.Lock()

        self.foreign_gpus = [i for i in range(self.num_devices) if is_gpu_in_use(i)]

        self.device_status = [i in self.foreign_gpus for i in range(self.num_devices)]

        self.jobs: Dict[str, TagJob] = {}
        # (media file, model) pairs, cannot have two jobs going on the same file with the same model
        self.files_tagging = set()
        
        self.cpu_available = threading.Event()
        self.cpu_available.set()

        # check if GPUs are available
        self.gpu_available = threading.Event()
        if any(status is False for status in self.device_status):
            self.gpu_available.set()

        # get all the "cpu slots" defined in the model config, and initialize to "available"
        self.cpuslots = {}
        for modelname, model_conf in config["services"].items():
            if model_conf.get("allowed_gpus", None) != []:
                # GPU models should not have cpu slots
                continue
            allowed_slots = model_conf.get("cpu_slots", [f"slot4{modelname}"])
            for slotname in allowed_slots:
                self.cpuslots[slotname] = False

    def run(self, feature: str, run_config: dict, files: List[str], allowed_gpus: List[int], allowed_cpus: List[str]) -> str:
        # Args:
        #     feature (str): The feature to tag the files with.
        #     run_config (dict): The configuration to run the model with. This is model-specific. Check the model's documentation.
        #     files (List[str]): The list of files to tag.
        # Returns:
        #     str: The job ID.
        # NOTE: this function creates a new thread to watch the job
        with self.lock:
            self.update_gpu_state()
            container = None
            files_added = []
            cpu_slot_to_use, gpu_device_to_use = None, None
            try:
                if len(allowed_gpus) == 0 and len(allowed_cpus) == 0:
                    raise NoResourceAvailable("No GPUs or CPU slots available")
                if len(allowed_gpus) > 0 and len(allowed_cpus) > 0:
                    raise ValueError("Cannot use both GPUs and CPU slots at the same time")
                
                if allowed_gpus:
                    gpu_device_to_use = self._acquire_gpu(allowed_gpus)
                    if gpu_device_to_use is None:
                        raise NoResourceAvailable("No GPUs available")
                else:
                    cpu_slot_to_use = self._acquire_cpu(allowed_cpus)
                    if cpu_slot_to_use is None:
                        raise NoResourceAvailable("No CPU slots available")
                    
                # start the container
                jobid = str(uuid.uuid4())
                if not os.path.exists(os.path.join(config["storage"]["logs"], feature)):
                    os.makedirs(os.path.join(config["storage"]["logs"], feature))
                logs_out = os.path.join(config["storage"]["logs"], feature, f"{jobid}.log")
                for f in files:
                    if (f, feature) in self.files_tagging:
                        raise ValueError(f"File {f} is already being tagged with {feature}")
                    self.files_tagging.add((f, feature))
                    files_added.append((f, feature))
                save_path = os.path.join(config["storage"]["tmp"], feature, jobid)
                container = create_container(self.client, feature, save_path, files, run_config, gpu_device_to_use, logs_out)
                container.start()
                self.jobs[jobid] = TagJob(container, feature, logs_out, gpu_device_to_use, cpu_slot_to_use, "Running", threading.Event(), time.time(), None, files, [], [], save_path=save_path)
            except Exception as e:
                # cleanup resources if job fails to start
                if container:
                    try:
                        self._stop_container(container)
                    except Exception as e2:
                        logger.error(f"Error while stopping container: feature={feature}: {e2}")
                        raise e2
                for f, feature in files_added:
                    self.files_tagging.remove((f, feature))
                if gpu_device_to_use is not None and gpu_device_to_use not in self.foreign_gpus:
                    self.device_status[gpu_device_to_use] = False
                    self.gpu_available.set()
                if cpu_slot_to_use is not None:
                    self.cpuslots[cpu_slot_to_use] = False
                    self.cpu_available.set()
                raise e
        threading.Thread(target=self._watch_job, args=(jobid, )).start()
        return jobid
    
    def _acquire_gpu(self, allowed_gpus: List[int]) -> Optional[int]:
        """
        Finds an available GPU from the list of allowed GPUs.
        Args:
            allowed_gpus (List[int]): The list of allowed GPUs to check.
        Returns:
            Optional[int]: The index of the available GPU, or None if no GPU is available.
            
        This function is not thread safe, and should be called with the lock held.
        """
        
        logger.debug("Trying to acquire gpu slot")

        self.update_gpu_state()

        gpu_device_to_use = None

        # check for an available GPU and set device statuses according
        for i, status in enumerate(self.device_status):
            if status:
                # gpu already in use
                continue
            if len(allowed_gpus) != 0 and i not in allowed_gpus:
                continue
            gpu_device_to_use = i
            self.device_status[i] = True
            if all(self.device_status[i] for i in range(self.num_devices)):
                self.gpu_available.clear()
            break
    
        return gpu_device_to_use

    def _acquire_cpu(self, allowed_cpu_slots: List[str]) -> Optional[str]:
        """
        Finds an available CPU slot from the list of allowed CPU slots.
        Args:
            allowed_cpu_slots (List[str]): The list of allowed CPU slots to check
        Returns:
            Optional[str]: The available cpu slot name, or None if none are available
            
        This function is not thread safe, and should be called with the lock held.
        """
        
        logger.debug("Trying to acquire cpu slot")

        cpu_slot_to_use = None

        # check for an available GPU and set device statuses according
        for slot, status in self.cpuslots.items():
            if status:
                # cpu slot already in use
                continue
            if slot not in allowed_cpu_slots:
                continue
            cpu_slot_to_use = slot
            self.cpuslots[slot] = True
            if all(self.cpuslots.values()):
                self.cpu_available.clear()

            break
    
        return cpu_slot_to_use

    # Not thread safe, must be called with lock
    def update_gpu_state(self):
        """
        Checks all GPUs that are not in use by the tagger and updates the device status if they are now available.
        """
        
        # check if any GPUs are freed by foreign processes ending
        for i in range(len(self.foreign_gpus)-1, -1, -1):
            gpu_idx = self.foreign_gpus[i]
            if not is_gpu_in_use(gpu_idx):
                self.foreign_gpus.pop(i)
                self.device_status[gpu_idx] = False
                self.gpu_available.set()
                
        # check if any GPUs are now in use by foreign processes
        # NOTE: if a foreign GPU starts on a GPU that is being used by a tagger, the tagger will continue to use the GPU and
        #                                                               this will not be detected until the tagger finishes.
        for i in range(self.num_devices):
            if not self.device_status[i] and is_gpu_in_use(i):
                self.foreign_gpus.append(i)
                self.device_status[i] = True
                
        if all(self.device_status[i] for i in range(self.num_devices)):
            self.gpu_available.clear()
            
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
        
        exit_code = job.container.attrs["State"]["ExitCode"]
        if exit_code != 0:
            logger.error(f"Job {jobid} failed to complete")
            with self.lock:
                self._cleanup_job(jobid, "Failed")
            return
        logger.info(f"Job {jobid} completed")

        tags = []
        for f in job.media_files:
            video_tags = os.path.join(job.save_path, f"{os.path.basename(f)}_tags.json")
            frame_tags = os.path.join(job.save_path, f"{os.path.basename(f)}_frametags.json")
            image_tags = os.path.join(job.save_path, f"{os.path.basename(f)}_imagetags.json")
            if os.path.exists(video_tags):
                tags.append(video_tags)
            if config["services"][job.feature].get("frame_level", False):
                if os.path.exists(frame_tags):
                    tags.append(frame_tags)
                if os.path.exists(image_tags):
                    tags.append(image_tags)
            
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

    def await_gpu(self, timeout: Optional[int]=None) -> bool:
        return self.gpu_available.wait(timeout=timeout)

    def await_cpu(self, timeout: Optional[int]=None) -> bool:
        return self.cpu_available.wait(timeout=timeout)

    # cleanup job resources
    # NOTE: NOT THREAD SAFE, must be called with lock
    def _cleanup_job(self, jobid: str, status: str) -> None:
        job = self.jobs[jobid]
        job.time_ended = time.time()
        job.status = status
        self._stop_container(job.container)
        for f in job.media_files:
            self.files_tagging.remove((f, job.feature))

        if job.device is not None:
            # free the gpu slot
            if is_gpu_in_use(job.device):
                # this implies a foreign process started while the job was running, so we don't free the gpu
                if job.device not in self.foreign_gpus:
                    self.foreign_gpus.append(job.device)
            else:
                self.device_status[job.device] = False
                self.gpu_available.set()
        else:
            # free the cpu slot
            self.cpuslots[job.cpu_slot] = False
            self.cpu_available.set()

    def _stop_container(self, container: Container) -> None:
        logger.info(f"Stopping container: status={container.status}")
        if container.status == "running":
            # podman client will kill if it doesn't stop within the timeout limit
            container.stop(timeout=5)
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
                return ResourceManager.TagJobStatus(job.status, tags=job.tags, \
                        time_elapsed=time.time() - job.time_started) 

    def shutdown(self):
        """
        Stop all running jobs and close the podman client.
        """
        for jobid, job in self.jobs.items():
            if job.status == "Running":
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
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        return bool(compute_procs or graphics_procs)
    except pynvml.NVMLError_Uninitialized as e:
        raise e