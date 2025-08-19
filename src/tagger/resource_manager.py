from collections import defaultdict
import time
import os
import uuid
import threading
from dataclasses import dataclass
from typing import List, Optional, Literal

import podman
from loguru import logger

from src.tagger.containers import TagContainer, create_container

from config import config

class NoResourceAvailable(Exception):
    pass

SystemResources = dict[str, int]

JobState = Literal["Queued", "Running", "Completed", "Failed", "Stopped"]

@dataclass
class JobStatus:
    status: JobState
    time_started: float
    time_ended: float | None
    error: Exception | None

@dataclass
class ContainerJob:
    container: TagContainer
    # resource requirements for the job
    reqs: SystemResources
    JobStatus: JobStatus
    stop_event: threading.Event
    gpu_indices: list[int]

@dataclass 
class sysConfig:
    # map gpu idx -> gpu type
    gpus: list[str]
    cpu_juice: int

def load_system_resouces(cfg: sysConfig) -> SystemResources:
    """
    Get currently available system resources
    """

    resources = defaultdict(int)

    for device_idx in range(len(cfg.gpus)):
        if cfg.gpus[device_idx] == "disabled":
            continue
        resources[cfg.gpus[device_idx]] += 1

    resources["cpu_juice"] = cfg.cpu_juice

    return dict(resources)

class SystemTagger:

    """
    Handles tagging jobs on internal resources.
    """

    def __init__(
            self, 
            cfg: sysConfig

    ):

        self.pclient = podman.PodmanClient()

        self.sys_config = cfg

        self.resource_lock = threading.Lock()
        self.resources = load_system_resouces(cfg)
        self.device_status = [False] * len(cfg.gpus)

        self.cond = threading.Condition()
        self.exit = threading.Event()

        
        self.jobs: dict[str, ContainerJob] = {}
        self.joblocks = defaultdict(threading.Lock)

        self.q = []

        self.starter = threading.Thread(target=self._start_jobs, daemon=True)
        self.starter.start()

        self.stopper = threading.Thread(target=self._stop_jobs, daemon=True)
        self.stopper.start()

    # TODO: add function to add new files to an existing job

    def start(
        self, 
        container: TagContainer,
        required_resources: SystemResources
    ) -> str:
        """
        Starts the tagging job and returns uuid
        """
        job_id = str(uuid.uuid4())
        job_status = JobStatus(status="Queued", time_started=time.time(), time_ended=None, error=None)
        self.jobs[job_id] = ContainerJob(container=container, reqs=required_resources, JobStatus=job_status, stop_event=threading.Event(), gpu_indices=[])
        with self.cond:
            self.q.append(job_id)
            self.cond.notify_all()

        return job_id

    def _can_start(self, job: ContainerJob) -> bool:

        with self.resource_lock:
            for resource, required in job.reqs.items():
                if self.resources.get(resource, 0) < required:
                    return False

        return True

    def _reserve_resources(self, job: ContainerJob) -> list[int]:
        """
        Reserves the resources for the job and returns the list of reserved GPU indices if any.

        Errors if resources are not available. This should not happen.
        """

        with self.resource_lock:
            gpu_resources = set(self.sys_config.gpus)

            reserved_gpus = []
            for resource, required in job.reqs.items():
                self.resources[resource] -= required
                assert self.resources[resource] >= 0
                if resource not in gpu_resources:
                    continue
                for _ in range(required):
                    for gpuidx, type in enumerate(self.sys_config.gpus):
                        if type == required and not self.device_status[gpuidx]:
                            # reserve the gpu
                            self.device_status[gpuidx] = True
                            reserved_gpus.append(gpuidx)
                            break
                assert len(reserved_gpus) == required

            return reserved_gpus

    def _run_job(self, jobid: str) -> None:

        with self.joblocks[jobid]:
            cj = self.jobs[jobid]
            if cj.JobStatus.time_ended:
                return

            gpu_indices = self._reserve_resources(cj)

            cj.JobStatus.status = "Running"
            cj.gpu_indices = gpu_indices

        try:
            if len(gpu_indices) > 1:
                # TODO: implement
                raise NotImplementedError("Multi-GPU tagging for one container is not implemented yet.")
            cj.container.start(self.pclient, gpu_indices[0])
        except Exception as e:
            self._stop_job(jobid, "Failed", e)

    def _stop_job(self, jobid: str, status: JobState, error: Exception | None = None) -> None:
        """
        Stops the job and cleans up resources.
        """

        with self.joblocks[jobid]:
            cj = self.jobs[jobid]

            if cj.JobStatus.time_ended:
                return

            if error:
                logger.exception(error)

            cj.JobStatus.status = status
            cj.JobStatus.error = error
            cj.JobStatus.time_ended = time.time()

            cj.container.stop()

        self._free_resources(cj)

        with self.cond:
            self.q.remove(cj)
            self.cond.notify_all()

    def _free_resources(self, cj: ContainerJob) -> None:
        """
        Frees the resources allocated for the job.
        """
        with self.resource_lock:
            for resource, required in cj.reqs.items():
                self.resources[resource] += required
                if resource == "gpu":
                    for gpu_idx in cj.gpu_indices:
                        self.device_status[gpu_idx] = False

    def _start_jobs(self):
        """
        Polls jobs from queue and starts them.
        """

        while not self.exit.is_set():
            with self.cond:
                pop_idx = None
                for i, jobid in enumerate(self.q):
                    cj = self.jobs[jobid]
                    if self._can_start(cj):
                        self._run_job(jobid)
                        pop_idx = i
                        break

                if pop_idx is not None:
                    self.q.pop(pop_idx)

                self.cond.wait()

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

                    tags = os.listdir(job.save_path)

                    if len(tags) > 0:
                        # remove the tag with the most recent write time
                        latest_idx = max(range(len(tags)), key=lambda i: os.path.getmtime(os.path.join(job.save_path, tags[i])))
                        tags.pop(latest_idx)
                        job.tags = [os.path.join(job.save_path, tag) for tag in tags]

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

    @dataclass
    class TagJobStatus:
        status: str # from TagJob.status
        tags: List[str]
        time_elapsed: Optional[float]

    def status(self, jobid: str) -> TagJobStatus:
        with self.lock:
            job = self.jobs[jobid]
            if job.time_ended is not None:
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