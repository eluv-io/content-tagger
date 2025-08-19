from collections import defaultdict
from concurrent.futures import thread
from copy import deepcopy
from shutil import copy
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
    jobstatus: JobStatus
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

        error = None
        with self.joblocks[jobid]:
            cj = self.jobs[jobid]
            if cj.jobstatus.time_ended:
                # user stopped it
                return

            gpu_indices = self._reserve_resources(cj)

            cj.jobstatus.status = "Running"
            cj.gpu_indices = gpu_indices

            try:
                if len(gpu_indices) > 1:
                    # TODO: implement
                    raise NotImplementedError("Multi-GPU tagging for one container is not implemented yet.")
                cj.container.start(self.pclient, gpu_indices[0])
            except Exception as e:
                error = e

        if error:
            self._stop_job(jobid, "Failed", error)

    def _stop_job(self, jobid: str, status: JobState, error: Exception | None = None) -> None:
        """
        Stops the job and cleans up resources.
        """

        with self.joblocks[jobid]:
            cj = self.jobs[jobid]

            if cj.jobstatus.time_ended:
                return

            if error:
                logger.exception(error)

            cj.jobstatus.status = status
            cj.jobstatus.error = error
            cj.jobstatus.time_ended = time.time()

            cj.container.stop()

        self._free_resources(cj)

        with self.cond:
            self.q.remove(cj)
            self.cond.notify_all()

    def stop(self, jobid: str) -> JobStatus:
        self._stop_job(jobid, "Stopped")
        return self.status(jobid)

    def status(self, jobid: str) -> JobStatus:
        with self.joblocks[jobid]:
            return deepcopy(self.jobs[jobid].jobstatus)

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
                self._clear_stopped_jobs()

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

    def _stop_jobs(self):
        while not self.exit.is_set():
            for jobid, job in self.jobs.items():
                if not job.jobstatus.status == "Running":
                    continue
                if job.container.is_running():
                    continue
                assert job.container.container is not None
                exit_code = job.container.container.attrs["State"]["ExitCode"]
                if exit_code != 0:
                    self._stop_job(jobid, "Failed", RuntimeError("Container encountered runtime error"))
                    return
                logger.info(f"Job {jobid} completed")
            time.sleep(2)

    def _clear_stopped_jobs(self) -> None:
        # clean the queue of stopped jobs
        newq = []
        for jobid in self.q:
            with self.joblocks[jobid]:
                if self.jobs[jobid].jobstatus.status == "Queued":
                    newq.append(jobid)
        self.q = newq