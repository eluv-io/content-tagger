from collections import defaultdict
from copy import deepcopy
import time
import uuid
import threading

from loguru import logger

from src.common.errors import MissingResourceError
from src.tag_containers.containers import TagContainer
from src.tagger.system_tagging.types import *

class SystemTagger:

    """
    Handles tagging jobs on internal resources.
    """

    def __init__(
        self, 
        cfg: SysConfig
    ):
        self.sys_config = cfg

        self.resource_lock = threading.Lock()
        self.active_resources = load_system_resources(cfg)
        self.gpu_status = [False] * len(cfg.gpus)

        self.total_resources = deepcopy(self.active_resources)

        self.cond = threading.Condition()
        self.q = []

        self.exit = threading.Event()

        self.joblocks = defaultdict(threading.Lock)
        self.jobs: dict[str, ContainerJob] = {}

        self.starter = threading.Thread(target=self._start_jobs, daemon=True)
        self.starter.start()

        self.stopper = threading.Thread(target=self._stop_jobs, daemon=True)
        self.stopper.start()

    # TODO: add function to add new files to an existing job

    def start(
        self, 
        container: TagContainer,
        required_resources: SystemResources,
        finished: threading.Event | None = None
    ) -> str:

        """
        Runs the container, bookkeeps system resources, and returns uuid
        """

        if not self._can_start(required_resources, self.total_resources):
            raise MissingResourceError("Insufficient resources available to start job on this system.")

        job_id = str(uuid.uuid4())
        job_status = JobStatus(status="Queued", time_started=time.time(), time_ended=None, error=None)
        with self.cond:
            self.jobs[job_id] = ContainerJob(container=container, reqs=required_resources, jobstatus=job_status, gpus_used=[], finished=finished)
            self.q.append(job_id)
            self.cond.notify_all()

        return job_id

    def shutdown(self) -> None:
        self.exit.set()
        self._terminate_containers()

    def _can_start(self, jobreqs: SystemResources, sys_resources: SystemResources) -> bool:

        """
        Can we start the container given the resources.
        """

        with self.resource_lock:
            for resr, req in jobreqs.items():
                if sys_resources.get(resr, 0) < req:
                    return False

        return True

    def _reserve_resources(self, job: ContainerJob) -> None:
        """
        Reserves the resources for the job and updates the job's GPU allocation.
        """

        with self.resource_lock:
            gpu_resources = set(self.sys_config.gpus)

            reserved_gpus = []
            for resr, req in job.reqs.items():
                self.active_resources[resr] -= req
                assert self.active_resources[resr] >= 0
                if resr not in gpu_resources:
                    continue
                for _ in range(req):
                    for gpuidx, type in enumerate(self.sys_config.gpus):
                        if type == resr and not self.gpu_status[gpuidx]:
                            self.gpu_status[gpuidx] = True
                            reserved_gpus.append(gpuidx)
                            break
                assert len(reserved_gpus) == req

            job.gpus_used = reserved_gpus

    def _run_job(self, jobid: str) -> None:

        error = None
        with self.joblocks[jobid]:
            cj = self.jobs[jobid]
            if cj.jobstatus.time_ended:
                # user stopped it
                return

            self._reserve_resources(cj)

            cj.jobstatus.status = "Running"

            try:
                if len(cj.gpus_used) > 1:
                    # TODO: implement
                    raise NotImplementedError("Multi-GPU tagging for one container is not implemented yet.")
                cj.container.start(cj.gpus_used[0] if cj.gpus_used else None)
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
            if jobid in self.q:
                self.q.remove(jobid)

        if cj.finished:
            cj.finished.set()

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

        gpu_resources = set(self.sys_config.gpus)

        with self.resource_lock:
            for resr, req in cj.reqs.items():
                self.active_resources[resr] += req
                if resr in gpu_resources:
                    for gpu_idx in cj.gpus_used:
                        self.gpu_status[gpu_idx] = False

        with self.cond:
            self.cond.notify_all()

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
                    if self._can_start(cj.reqs, self.active_resources):
                        self._run_job(jobid)
                        pop_idx = i
                        break

                if pop_idx is not None:
                    self.q.pop(pop_idx)

                self.cond.wait()

    def _stop_jobs(self):

        """
        Checks for jobs which have stopped and cleans up resources.
        """

        while not self.exit.is_set():
            with self.cond:
                for jobid, job in self.jobs.items():
                    if job.jobstatus.status != "Running":
                        continue
                    if job.container.is_running():
                        continue
                    assert job.container.container is not None
                    exit_code = job.container.container.attrs["State"]["ExitCode"]
                    if exit_code != 0:
                        self._stop_job(jobid, "Failed", RuntimeError("Container encountered runtime error"))
                        continue
                    else:
                        self._stop_job(jobid, "Completed")
                    logger.info(f"Job {jobid} completed")
            time.sleep(2)

    def _terminate_containers(self):
        for job in self.jobs.values():
            if job.container.is_running():
                job.container.stop()

    def _clear_stopped_jobs(self) -> None:
        # clean the queue of stopped jobs
        newq = []
        for jobid in self.q:
            with self.joblocks[jobid]:
                if self.jobs[jobid].jobstatus.status == "Queued":
                    newq.append(jobid)
        self.q = newq

def load_system_resources(cfg: SysConfig) -> SystemResources:
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