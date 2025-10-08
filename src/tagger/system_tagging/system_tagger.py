from collections import defaultdict
from copy import deepcopy
import time
import uuid
import threading
import queue

from src.common.errors import BadRequestError
from src.tag_containers.containers import TagContainer
from src.common.model import SystemResources
from src.tagger.system_tagging.model import *
from src.tagger.system_tagging.state import *
from src.tagger.system_tagging.message_types import *
from src.common.logging import logger

logger = logger.bind(name="System Tagger")

class SystemTagger:
    """
    Handles tagging jobs on internal resources using single-thread actor pattern.
    """

    def __init__(self, cfg: SysConfig):
        self.sys_config = cfg
        
        # --- Actor state (only accessed by actor thread) --- 
        self.resource_state = self._load_system_resources(cfg)
        # maps jobid (internal uuid) -> ContainerJob
        self.jobs: dict[str, ContainerJob] = {}
        # queue of jobids, should be in self.jobs
        self.job_queue: list[str] = []
        self.exit_requested = False

        self.mailbox = queue.Queue()
        
        # Actor thread
        self.actor_thread = threading.Thread(target=self._actor_loop, daemon=True)
        self.actor_thread.start()
        
        # Container monitor thread (notifies when containers finish)
        self.monitor_thread = threading.Thread(target=self._monitor_containers, daemon=True)
        self.monitor_thread.start()

    def start(
        self, 
        container: TagContainer,
        finished: threading.Event | None = None
    ) -> str:
        """Starts a job by enqueueing the container, and returns uuid for the job.
        
        Args:
            container: TagContainer to run
            finished: Optional threading.Event. If provided, this will be set when job completes or fails.
                Useful for triggering downstream tasks.
        """

        required_resources = container.required_resources()
        
        # Quick capacity check before queuing
        if not self.check_capacity(required_resources):
            raise BadRequestError("Insufficient resources available to start job on this system.")
        
        logger.info("Received request to start container", extra={
            "container": container.name(),
            "resources": required_resources
        })

        request = StartJobRequest(
            container=container,
            finished=finished
        )
        
        response_queue = queue.Queue()
        message = Message(MessageType.START_JOB, {"request": request}, response_queue)
        self.mailbox.put(message)
        
        return response_queue.get()

    # TODO: This doesn't need to go through the actor thread.
    def check_capacity(self, required_resources: SystemResources) -> bool:
        """Checks if the system has enough resources to start a job."""
        response_queue = queue.Queue()
        message = Message(MessageType.CHECK_CAPACITY, {"resources": required_resources}, response_queue)
        self.mailbox.put(message)
        return response_queue.get()

    def stop(self, jobid: str) -> ContainerJobStatus | None:
        """Stop a job and return its status. Return None if jobid not found."""
        logger.info("Received request to stop job", extra={"jobid": jobid[:8]})
        request = StopJobRequest(jobid=jobid, status="Stopped")
        response_queue = queue.Queue()
        message = Message(MessageType.STOP_JOB, {"request": request}, response_queue)
        self.mailbox.put(message)
        return response_queue.get()

    def status(self, jobid: str) -> ContainerJobStatus:
        """Get job status."""
        logger.bind(extra={"jobid": jobid[:8]}).info("Received request to get status for job")
        response_queue = queue.Queue()
        message = Message(MessageType.GET_STATUS, {"jobid": jobid}, response_queue)
        self.mailbox.put(message)
        return response_queue.get()

    def shutdown(self) -> None:
        """Shuts down the actor and all running jobs."""
        logger.info("Shutdown requested for SystemTagger")
        response_queue = queue.Queue()
        message = Message(MessageType.SHUTDOWN, {}, response_queue)
        self.mailbox.put(message)
        response_queue.get()

    def _actor_loop(self):
        """Processes all messages sequentially and starts queued jobs."""
        last_status_log = time.time()
        status_log_interval = 30.0
        
        while not self.exit_requested:
            try:
                # Process messages with timeout to allow periodic queue processing
                try:
                    message = self.mailbox.get(timeout=0.1)
                    self._handle_message(message)
                except queue.Empty:
                    pass
                
                # Try to start queued jobs
                self._process_job_queue()
                
                current_time = time.time()
                if current_time - last_status_log >= status_log_interval:
                    if not self.mailbox.empty():
                        self._log_queue_status()
                    else:
                        logger.debug("No new messages")
                    last_status_log = current_time

            except Exception as e:
                logger.opt(exception=e).error("Error in actor loop")

        logger.info("Actor thread shutting down")

    def _log_queue_status(self):
        """Log current queue and job status"""
        message_queue_size = self.mailbox.qsize()
        job_queue_size = len(self.job_queue)
        
        # Count jobs by status
        job_counts = {}
        for job in self.jobs.values():
            status = job.jobstatus.status
            job_counts[status] = job_counts.get(status, 0) + 1
        
        # Format resource usage
        total_gpus = len(self.sys_config.gpus)
        used_gpus = sum(1 for used in self.resource_state.gpu_status if used)
        
        logger.info(
            f"Queue Status - Messages: {message_queue_size}, "
            f"Job Queue: {job_queue_size}, "
            f"Jobs: {dict(job_counts)}, "
            f"GPUs: {used_gpus}/{total_gpus}, "
            f"Resources: {dict(self.resource_state.available)}"
        )
        
        # Log individual queued jobs if any
        if self.job_queue:
            queued_jobs = []
            for jobid in self.job_queue:
                if jobid in self.jobs:
                    job = self.jobs[jobid]
                    queued_jobs.append(f"{jobid[:8]}:{job.container.name()}")
            logger.debug(f"Queued jobs: {queued_jobs}")

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
            jobstatus=ContainerJobStatus.starting(),
            gpus_used=[],
            finished=request.finished
        )
        
        self.jobs[job_id] = job
        self.job_queue.append(job_id)
        
        logger.info("Job queued", extra={
            "jobid": job_id,
            "container": request.container.name(),
        })

        assert message.response_queue is not None
        message.response_queue.put(job_id)

    def _handle_stop_job(self, message: Message):
        request: StopJobRequest = message.data["request"]

        if request.jobid not in self.jobs:
            logger.warning("Stop requested for unknown job", extra={"jobid": request.jobid})
            if message.response_queue:
                message.response_queue.put(None)
            return
        
        job = self.jobs[request.jobid]

        log_fields = {"jobid": request.jobid[:8], "container": job.container.name(), "status": job.jobstatus.status}
        
        if job.jobstatus.time_ended:
            logger.warning("stop requested for already stopped job", extra=log_fields)
            if message.response_queue:
                message.response_queue.put(deepcopy(job.jobstatus))
            return
        
        if request.error:
            logger.opt(exception=request.error).info("stopping job with error", extra=log_fields)
        else:
            logger.info("stopping job", extra=log_fields)
        
        # TODO: stop in separate thread to not block
        try:
            job.container.stop()
        except Exception as e:
            logger.opt(exception=e).error("Error stopping container", extra=log_fields)
        
        if job.jobstatus.status == "Running":
            self._free_resources(job)
            if request.jobid in self.job_queue:
                self.job_queue.remove(request.jobid)
            logger.info("successfully stopped running job and freed resources", extra={**log_fields, "new_available_resources": self.resource_state.available})
        elif job.jobstatus.status == "Queued":
            assert request.jobid in self.job_queue
            self.job_queue.remove(request.jobid)
            logger.info("successfully removed job from queue", extra=log_fields)
        else:
            logger.error("Job in unexpected state during stop", extra=log_fields)
            raise RuntimeError("Job in unexpected state during stop")

        job.jobstatus.status = request.status
        job.jobstatus.error = request.error
        job.jobstatus.time_ended = time.time()
        
        if job.finished:
            job.finished.set()
        
        if message.response_queue:
            message.response_queue.put(deepcopy(job.jobstatus))

    def _handle_check_capacity(self, message: Message):
        required_resources = message.data["resources"]
        can_start = self._can_start(required_resources, self.resource_state.total)
        assert message.response_queue is not None
        message.response_queue.put(can_start)

    def _handle_get_status(self, message: Message):
        jobid = message.data["jobid"]
        log_fields = {"jobid": jobid[:8]}
        if jobid in self.jobs:
            log_fields["container"] = self.jobs[jobid].container.name()
            status = deepcopy(self.jobs[jobid].jobstatus)
        else:
            logger.warning("Status requested for unknown job", extra=log_fields)
            status = None
        logger.info("Returning job status", extra=log_fields)
        assert message.response_queue is not None
        message.response_queue.put(status)

    def _handle_shutdown(self, message: Message):
        logger.info("shutdown requested")
        self._terminate_all_jobs()
        self.exit_requested = True
        if message.response_queue:
            message.response_queue.put(True)

    def _handle_container_finished(self, message: Message):
        jobid = message.data["jobid"]
        exit_code = message.data["exit_code"]
        
        if jobid not in self.jobs:
            logger.warning("Received finished message for unknown job", extra={"jobid": jobid[:8]})
            return
        
        job = self.jobs[jobid]

        log_fields = {"jobid": jobid[:8], "container": job.container.name(), "exit_code": exit_code}
        if job.jobstatus.time_ended:
            logger.warning("Received finished message for already finished job", extra=log_fields)
            return
        
        if exit_code == 0:
            logger.info("container finished successfully", extra=log_fields)
            self._handle_stop_job(Message(
                MessageType.STOP_JOB, 
                {"request": StopJobRequest(jobid, "Completed")}
            ))
        else:
            error = RuntimeError(f"Container {job.container.name()} exited with code {exit_code}")
            logger.info("tagging container failed, stopping job with error status", extra=log_fields)
            self._handle_stop_job(Message(
                MessageType.STOP_JOB,
                {"request": StopJobRequest(jobid, "Failed", error)}
            ))

    def _process_job_queue(self):
        """Try to start queued jobs."""
        if not self.job_queue:
            return

        # Try to start jobs
        for jobid in self.job_queue[:]:
            job = self.jobs[jobid]

            assert job.jobstatus.status == "Queued"

            container_reqs = job.container.required_resources()
            if self._can_start(container_reqs, self.resource_state.available):
                self._start_job(jobid)
                if jobid in self.job_queue:
                    # we need the check because _start_job may have removed it already if it errors
                    self.job_queue.remove(jobid)

    def _start_job(self, jobid: str) -> None:
        job = self.jobs[jobid]
        
        if job.jobstatus.time_ended:
            logger.warning("Attempted to start already finished job", extra={"jobid": jobid[:8], "container": job.container.name()})
            return
        
        # Reserve resources
        self._reserve_resources(job)
        
        job.jobstatus.status = "Running"
        
        logger.info(f"Starting job {jobid}: {job.container.name()}")

        # Start container
        try:
            if len(job.gpus_used) > 1:
                raise NotImplementedError("Multi-GPU tagging not implemented")

            gpu = job.gpus_used[0] if job.gpus_used else None
            job.container.start(gpu)
        except Exception as e:
            logger.error(f"Failed to start job {jobid}\n{job.container.name()}: {e}")
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

        job_reqs = job.container.required_resources()
        
        for resr, req in job_reqs.items():
            self.resource_state.available[resr] -= req
            assert self.resource_state.available[resr] >= 0, f"Resource {resr} went negative"
            
            if resr not in gpu_resources:
                continue
                
            # Allocate GPUs
            for _ in range(req):
                for gpuidx, gpu_type in enumerate(self.sys_config.gpus):
                    if gpu_type == resr and not self.resource_state.gpu_status[gpuidx]:
                        self.resource_state.gpu_status[gpuidx] = True
                        reserved_gpus.append(gpuidx)
                        break
                else:
                    # This shouldn't happen if _can_start worked correctly
                    raise RuntimeError(f"Could not allocate GPU for resource {resr}")
        
        job.gpus_used = reserved_gpus

    def _free_resources(self, job: ContainerJob) -> None:
        """Free resources allocated to a job."""
        gpu_resources = set(self.sys_config.gpus)
        job_reqs = job.container.required_resources()
        
        for resr, req in job_reqs.items():
            self.resource_state.available[resr] += req
            
            if resr in gpu_resources:
                for gpu_idx in job.gpus_used:
                    self.resource_state.gpu_status[gpu_idx] = False
        
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
        """Monitor running containers and notifies when they finish."""
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
                        self.mailbox.put(message)
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error in container monitor: {e}")
                time.sleep(1.0)
        logger.info("Container monitor shutting down")

    def _load_system_resources(self, cfg: SysConfig) -> ResourceState:
        """Get currently available system resources and load the state object for tracking what's available"""
        resources = defaultdict(int)
        
        for device_idx in range(len(cfg.gpus)):
            if cfg.gpus[device_idx] == "disabled":
                continue
            resources[cfg.gpus[device_idx]] += 1

        gpu_resources = set(resources)
        other_resources = set(cfg.resources)

        # should not be overlap
        assert len(gpu_resources) + len(other_resources) == len(gpu_resources | other_resources)

        resources.update(cfg.resources)

        logger.debug(f"System resources: {dict(resources)}")

        return ResourceState(
            total=SystemResources(dict(resources)),
            available=SystemResources(dict(resources)),
            gpu_status=[False] * len(cfg.gpus)
        )