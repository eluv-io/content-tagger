from collections import defaultdict
from copy import deepcopy
import time
import uuid
import threading
import queue
from enum import Enum
from dataclasses import dataclass

from loguru import logger

from src.common.errors import BadRequestError
from src.common.resources import SystemResources
from src.tag_containers.containers import TagContainer
from src.tagger.system_tagging.types import *

class MessageType(Enum):
    START_JOB = "start_job"
    STOP_JOB = "stop_job"
    CHECK_CAPACITY = "check_capacity"
    GET_STATUS = "get_status"
    SHUTDOWN = "shutdown"
    CONTAINER_FINISHED = "container_finished"

@dataclass
class Message:
    type: MessageType
    data: dict
    response_queue: queue.Queue | None = None

@dataclass
class StartJobRequest:
    container: TagContainer
    required_resources: SystemResources
    finished: threading.Event | None = None

@dataclass
class StopJobRequest:
    jobid: str
    status: JobStateDescription
    error: Exception | None = None

class SystemTagger:
    """
    Handles tagging jobs on internal resources using single-thread actor pattern.
    """

    def __init__(self, cfg: SysConfig):
        self.sys_config = cfg
        
        # Actor state (only accessed by actor thread)
        self.active_resources = load_system_resources(cfg)
        self.gpu_status = [False] * len(cfg.gpus)
        self.total_resources = deepcopy(self.active_resources)
        self.jobs: dict[str, ContainerJob] = {}
        self.job_queue: list[str] = []
        self.exit_requested = False
        
        # Communication with actor
        self.message_queue = queue.Queue()
        
        # Actor thread
        self.actor_thread = threading.Thread(target=self._actor_loop, daemon=True)
        self.actor_thread.start()
        
        # Container monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_containers, daemon=True)
        self.monitor_thread.start()

    def start(
        self, 
        container: TagContainer,
        required_resources: SystemResources,
        finished: threading.Event | None = None
    ) -> str:
        """Starts a job by enqueueing the container, and returns uuid for the job."""
        
        # Quick capacity check before queuing
        if not self.check_capacity(required_resources):
            raise BadRequestError("Insufficient resources available to start job on this system.")
        
        request = StartJobRequest(
            container=container,
            required_resources=required_resources,
            finished=finished
        )
        
        response_queue = queue.Queue()
        message = Message(MessageType.START_JOB, {"request": request}, response_queue)
        self.message_queue.put(message)
        
        return response_queue.get()

    def check_capacity(self, required_resources: SystemResources) -> bool:
        """Checks if the system has enough resources to start a job."""
        response_queue = queue.Queue()
        message = Message(MessageType.CHECK_CAPACITY, {"resources": required_resources}, response_queue)
        self.message_queue.put(message)
        return response_queue.get()

    def stop(self, jobid: str) -> ContainerJobStatus | None:
        """Stop a job and return its status. Return None if jobid not found."""
        request = StopJobRequest(jobid=jobid, status="Stopped")
        response_queue = queue.Queue()
        message = Message(MessageType.STOP_JOB, {"request": request}, response_queue)
        self.message_queue.put(message)
        return response_queue.get()

    def status(self, jobid: str) -> ContainerJobStatus:
        """Get job status."""
        response_queue = queue.Queue()
        message = Message(MessageType.GET_STATUS, {"jobid": jobid}, response_queue)
        self.message_queue.put(message)
        return response_queue.get()

    def shutdown(self) -> None:
        response_queue = queue.Queue()
        message = Message(MessageType.SHUTDOWN, {}, response_queue)
        self.message_queue.put(message)
        response_queue.get()

    def _actor_loop(self):
        """Processes all messages sequentially."""
        while not self.exit_requested:
            try:
                # Process messages with timeout to allow periodic queue processing
                try:
                    message = self.message_queue.get(timeout=0.1)
                    self._handle_message(message)
                except queue.Empty:
                    pass
                
                # Try to start queued jobs
                self._process_job_queue()
                
            except Exception as e:
                logger.exception(f"Error in actor loop: {e}")
        
        logger.info("SystemTagger actor shutting down")

    def _handle_message(self, message: Message):
        if message.type == MessageType.START_JOB:
            self._handle_start_job(message)
        elif message.type == MessageType.STOP_JOB:
            self._handle_stop_job(message)
        elif message.type == MessageType.CHECK_CAPACITY:
            self._handle_check_capacity(message)
        elif message.type == MessageType.GET_STATUS:
            self._handle_get_status(message)
        elif message.type == MessageType.SHUTDOWN:
            self._handle_shutdown(message)
        elif message.type == MessageType.CONTAINER_FINISHED:
            self._handle_container_finished(message)
        else:
            logger.warning(f"Unknown message type: {message.type}")

    def _handle_start_job(self, message: Message):
        request: StartJobRequest = message.data["request"]

        job_id = str(uuid.uuid4())
        job = ContainerJob(
            container=request.container,
            reqs=request.required_resources,
            jobstatus=ContainerJobStatus.starting(),
            gpus_used=[],
            finished=request.finished
        )
        
        self.jobs[job_id] = job
        self.job_queue.append(job_id)
        
        logger.info(f"Job {job_id} queued")
        assert message.response_queue is not None
        message.response_queue.put(job_id)

    def _handle_stop_job(self, message: Message):
        request: StopJobRequest = message.data["request"]

        if request.jobid not in self.jobs:
            logger.warning(f"Job {request.jobid} not found")
            if message.response_queue:
                message.response_queue.put(None)
            return
        
        job = self.jobs[request.jobid]
        
        if job.jobstatus.time_ended:
            # Already stopped
            if message.response_queue:
                message.response_queue.put(deepcopy(job.jobstatus))
            return
        
        logger.info(f"Stopping job {request.jobid}")
        
        if request.error:
            logger.exception(f"Job {request.jobid} error: {request.error}")
        
        job.jobstatus.status = request.status
        job.jobstatus.error = request.error
        job.jobstatus.time_ended = time.time()
        
        # TODO: stop in separate thread to not block
        try:
            job.container.stop()
        except Exception as e:
            logger.error(f"Failed to stop container for job {request.jobid}: {e}")
        
        self._free_resources(job)
        
        if request.jobid in self.job_queue:
            self.job_queue.remove(request.jobid)
        
        if job.finished:
            job.finished.set()
        
        if message.response_queue:
            message.response_queue.put(deepcopy(job.jobstatus))

    def _handle_check_capacity(self, message: Message):
        required_resources = message.data["resources"]
        can_start = self._can_start(required_resources, self.total_resources)
        assert message.response_queue is not None
        message.response_queue.put(can_start)

    def _handle_get_status(self, message: Message):
        jobid = message.data["jobid"]
        if jobid in self.jobs:
            status = deepcopy(self.jobs[jobid].jobstatus)
        else:
            status = None
        assert message.response_queue is not None
        message.response_queue.put(status)

    def _handle_shutdown(self, message: Message):
        logger.info("Shutdown requested")
        self._terminate_all_jobs()
        self.exit_requested = True
        if message.response_queue:
            message.response_queue.put(True)

    def _handle_container_finished(self, message: Message):
        jobid = message.data["jobid"]
        exit_code = message.data["exit_code"]
        
        if jobid not in self.jobs:
            return
        
        job = self.jobs[jobid]
        if job.jobstatus.time_ended:
            return  # Already stopped
        
        if exit_code == 0:
            self._handle_stop_job(Message(
                MessageType.STOP_JOB, 
                {"request": StopJobRequest(jobid, "Completed")}
            ))
        else:
            error = RuntimeError(f"Container exited with code {exit_code}")
            self._handle_stop_job(Message(
                MessageType.STOP_JOB,
                {"request": StopJobRequest(jobid, "Failed", error)}
            ))

    def _process_job_queue(self):
        """Try to start queued jobs."""
        if not self.job_queue:
            return
        
        # Clean queue of stopped jobs first
        self.job_queue = [
            jobid for jobid in self.job_queue 
            if jobid in self.jobs and self.jobs[jobid].jobstatus.status == "Queued"
        ]
        
        # Try to start jobs
        for jobid in self.job_queue[:]:
            job = self.jobs[jobid]
            
            if self._can_start(job.reqs, self.active_resources):
                self._start_job(jobid)
                if jobid in self.job_queue:
                    # we need the check because _start_job may have removed it already if it errors
                    self.job_queue.remove(jobid)

    def _start_job(self, jobid: str) -> None:
        job = self.jobs[jobid]
        
        if job.jobstatus.time_ended:
            logger.warning(f"Attempted to start already finished job {jobid}")
            return
        
        # Reserve resources
        self._reserve_resources(job)
        
        # Update status
        job.jobstatus.status = "Running"
        
        logger.info(f"Starting job {jobid}")

        # Start container
        try:
            if len(job.gpus_used) > 1:
                raise NotImplementedError("Multi-GPU tagging not implemented")

            gpu = job.gpus_used[0] if job.gpus_used else None
            job.container.start(gpu)
        except Exception as e:
            logger.error(f"Failed to start job {jobid}: {e}")
            self._handle_stop_job(Message(
                MessageType.STOP_JOB,
                {"request": StopJobRequest(jobid, "Failed", e)}
            ))

    def _can_start(self, jobreqs: SystemResources, sys_resources: SystemResources) -> bool:
        """Check if job can start with current resources."""
        for resr, req in jobreqs.items():
            if sys_resources.get(resr, 0) < req:
                return False
        return True

    def _reserve_resources(self, job: ContainerJob) -> None:
        """Reserve resources for a job."""
        gpu_resources = set(self.sys_config.gpus)
        reserved_gpus = []
        
        for resr, req in job.reqs.items():
            self.active_resources[resr] -= req
            assert self.active_resources[resr] >= 0, f"Resource {resr} went negative"
            
            if resr not in gpu_resources:
                continue
                
            # Allocate GPUs
            for _ in range(req):
                for gpuidx, gpu_type in enumerate(self.sys_config.gpus):
                    if gpu_type == resr and not self.gpu_status[gpuidx]:
                        self.gpu_status[gpuidx] = True
                        reserved_gpus.append(gpuidx)
                        break
                else:
                    # This shouldn't happen if _can_start worked correctly
                    raise RuntimeError(f"Could not allocate GPU for resource {resr}")
        
        job.gpus_used = reserved_gpus

    def _free_resources(self, job: ContainerJob) -> None:
        """Free resources allocated to a job."""
        gpu_resources = set(self.sys_config.gpus)
        
        for resr, req in job.reqs.items():
            self.active_resources[resr] += req
            
            if resr in gpu_resources:
                for gpu_idx in job.gpus_used:
                    self.gpu_status[gpu_idx] = False
        
        job.gpus_used = []

    def _terminate_all_jobs(self):
        """Terminate all active jobs."""
        logger.info("Terminating all jobs")
        
        active_jobids = [
            jobid for jobid, job in self.jobs.items()
            if not job.jobstatus.time_ended
        ]
        
        for jobid in active_jobids:
            self._handle_stop_job(Message(
                MessageType.STOP_JOB,
                {"request": StopJobRequest(jobid, "Stopped")}
            ))

    def _monitor_containers(self):
        """Monitor running containers and notify when they finish."""
        while not self.exit_requested:
            try:
                running_jobs = [
                    (jobid, job) for jobid, job in self.jobs.items()
                    if job.jobstatus.status == "Running" and not job.jobstatus.time_ended
                ]
                
                for jobid, job in running_jobs:
                    exit_code = job.container.exit_code()
                    if exit_code is not None:    
                        # Send message to actor
                        message = Message(
                            MessageType.CONTAINER_FINISHED,
                            {"jobid": jobid, "exit_code": exit_code}
                        )
                        self.message_queue.put(message)
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error in container monitor: {e}")
                time.sleep(1.0)
        logger.info("Container monitor shutting down")

def load_system_resources(cfg: SysConfig) -> SystemResources:
    """Get currently available system resources"""
    resources = defaultdict(int)
    
    for device_idx in range(len(cfg.gpus)):
        if cfg.gpus[device_idx] == "disabled":
            continue
        resources[cfg.gpus[device_idx]] += 1
    
    resources["cpu_juice"] = cfg.cpu_juice
    
    return dict(resources)