import pytest
import time
import threading
from unittest.mock import Mock

from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tagger.system_tagging.types import SysConfig


class MockTagContainer:
    """Mock TagContainer that simulates work"""
    
    def __init__(self, work_duration: float = 0.1):
        self.work_duration = work_duration
        self.is_started = False
        self.is_stopped = False
        self.stop_called = False
        self.container = Mock()
        self.container.attrs = {"State": {"ExitCode": 0}}
        
    def start(self, gpu_idx: int | None = None) -> None:
        self.is_started = True
        self.gpu_idx = gpu_idx
        # Simulate work in background
        def work():
            time.sleep(self.work_duration)
            self.is_stopped = True
        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        self.stop_called = True
        self.is_stopped = True

    def is_running(self) -> bool:
        return self.is_started and not self.is_stopped
    
    def exit_code(self) -> int | None:
        if self.is_stopped:
            return 0
        return None
    
    def name(self) -> str:
        return "MockContainer"


@pytest.fixture
def sys_config():
    """Create a test system configuration"""
    return SysConfig(
        gpus=["A6000", "A6000", "disabled"],  # 2 A6000 GPUs, 1 disabled
        resources={"cpu_juice": 4}
    )


@pytest.fixture
def system_tagger(sys_config):
    """Create a SystemTagger instance for testing"""
    tagger = SystemTagger(sys_config)
    yield tagger
    if tagger.exit_requested is False:
        tagger.shutdown()


def test_start_job_with_sufficient_resources(system_tagger):
    """Test starting a job when resources are available"""
    container = MockTagContainer()
    resources = {"A6000": 1, "cpu_juice": 2}
    
    job_id = system_tagger.start(container, resources)
    
    assert job_id is not None
    assert isinstance(job_id, str)
    
    # Check initial status
    status = system_tagger.status(job_id)
    assert status.status in ["Queued", "Running"]
    assert status.time_started > 0
    assert status.time_ended is None
    assert status.error is None


def test_start_job_with_insufficient_resources(system_tagger):
    """Test starting a job when resources are not available"""
    container = MockTagContainer()
    resources = {"cpu_juice": 5}  # More juice than available
    
    with pytest.raises(Exception):
        system_tagger.start(container, resources)


def test_job_completion_flow(system_tagger):
    """Test that a job goes through the complete lifecycle"""
    container = MockTagContainer(work_duration=0.1)
    resources = {"A6000": 1}
    
    job_id = system_tagger.start(container, resources)
    
    # Wait for job to complete
    timeout = 5.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = system_tagger.status(job_id)
        if status.status == "Completed":
            break
        time.sleep(0.05)
    
    final_status = system_tagger.status(job_id)
    assert final_status.status == "Completed"
    assert final_status.time_ended is not None
    assert final_status.time_ended > final_status.time_started


def test_stop_job(system_tagger):
    """Test manually stopping a job"""
    container = MockTagContainer(work_duration=1.0)  # Long duration
    resources = {"A6000": 1}
    
    job_id = system_tagger.start(container, resources)
    
    # Let it start
    time.sleep(0.1)
    
    # Stop the job
    stop_status = system_tagger.stop(job_id)
    assert stop_status.status == "Stopped"
    assert stop_status.time_ended is not None
    
    # Verify status is consistent
    status = system_tagger.status(job_id)
    assert status.status == "Stopped"


def test_multiple_jobs_queuing(system_tagger):
    """Test that jobs are queued when resources are exhausted"""
    containers = [MockTagContainer(work_duration=0.2) for _ in range(3)]
    resources = {"A6000": 1}  # Each job needs 1 GPU, but we only have 2
    
    job_ids = []
    for container in containers:
        job_id = system_tagger.start(container, resources)
        job_ids.append(job_id)
    
    # Check that some jobs are queued
    queued_count = 0
    running_count = 0
    for job_id in job_ids:
        status = system_tagger.status(job_id)
        if status.status == "Queued":
            queued_count += 1
        elif status.status == "Running":
            running_count += 1
    
    # Should have at least one queued job since we have more jobs than GPUs
    assert queued_count >= 1
    assert running_count <= 2  # Can't run more than available GPUs


def test_job_queue_processing(system_tagger):
    """Test that queued jobs get started when resources become available"""
    containers = [MockTagContainer(work_duration=0.1) for _ in range(3)]
    resources = {"A6000": 1}
    
    job_ids = []
    for container in containers:
        job_id = system_tagger.start(container, resources)
        job_ids.append(job_id)
    
    # Wait for all jobs to complete
    timeout = 2
    start_time = time.time()
    while time.time() - start_time < timeout:
        completed_count = 0
        for job_id in job_ids:
            status = system_tagger.status(job_id)
            if status.status == "Completed":
                completed_count += 1
        
        if completed_count == len(job_ids):
            break
        time.sleep(0.2)
    
    # All jobs should eventually complete
    for job_id in job_ids:
        status = system_tagger.status(job_id)
        print(status)
        assert status.status == "Completed"


def test_resource_allocation_and_cleanup(system_tagger):
    """Test that resources are properly allocated and cleaned up"""
    container = MockTagContainer(work_duration=0.1)
    resources = {"A6000": 1, "cpu_juice": 2}
    
    # Check initial resources
    initial_resources = system_tagger.active_resources.copy()
    
    job_id = system_tagger.start(container, resources)

    # Resources should be allocated
    mid_resources = system_tagger.active_resources
    assert mid_resources["A6000"] == initial_resources["A6000"] - 1
    assert mid_resources["cpu_juice"] == initial_resources["cpu_juice"] - 2
    
    # Wait for job to complete
    timeout = 2.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = system_tagger.status(job_id)
        if status.status == "Completed":
            break
        time.sleep(0.05)
    
    # Resources should be restored
    final_resources = system_tagger.active_resources
    assert final_resources == initial_resources


def test_shutdown_stops_all_jobs(system_tagger):
    """Test that shutdown properly stops all running jobs"""
    containers = [MockTagContainer(work_duration=1.0) for _ in range(2)]
    resources = {"A6000": 1}
    
    job_ids = []
    for container in containers:
        job_id = system_tagger.start(container, resources)
        job_ids.append(job_id)
    
    # Let jobs start
    time.sleep(0.1)
    
    # Shutdown
    system_tagger.shutdown()
    
    # Verify containers are stopped
    for container in containers:
        assert container.stop_called


def test_finished_event_notification(system_tagger):
    """Test that the finished event is properly set when job completes"""
    container = MockTagContainer(work_duration=0.1)
    resources = {"A6000": 1}
    finished_event = threading.Event()
    
    job_id = system_tagger.start(container, resources, finished_event)
    
    # Wait for the event to be set
    assert finished_event.wait(timeout=2.0), "Finished event was not set"
    
    # Job should be completed
    status = system_tagger.status(job_id)
    assert status.status == "Completed"


def test_job_failure_handling(system_tagger):
    """Test handling of job failures"""
    container = MockTagContainer()
    
    # Mock the container to raise an exception on start
    def failing_start(gpu_idx=None):
        raise RuntimeError("Container failed to start")
    
    container.start = failing_start
    resources = {"A6000": 1}
    
    job_id = system_tagger.start(container, resources)
    
    # Wait for failure to be detected
    timeout = 2.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = system_tagger.status(job_id)
        if status.status == "Failed":
            break
        time.sleep(0.05)
    
    status = system_tagger.status(job_id)
    assert status.status == "Failed"
    assert status.error is not None


def test_gpu_allocation(system_tagger):
    """Test that GPU allocation works correctly"""
    container = MockTagContainer(work_duration=0.1)
    resources = {"A6000": 1}
    
    job_id = system_tagger.start(container, resources)
    
    # Wait for job to start
    time.sleep(0.1)
    
    # Check that GPU was allocated
    job = system_tagger.jobs[job_id]
    if job.jobstatus.status == "Running":
        assert len(job.gpus_used) == 1
        assert container.gpu_idx is not None


def test_cpu_only_job(system_tagger):
    """Test running a job that only needs CPU resources"""
    container = MockTagContainer(work_duration=0.1)
    resources = {"cpu_juice": 2}  # No GPU required
    
    job_id = system_tagger.start(container, resources)
    
    # Wait for completion
    timeout = 2.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = system_tagger.status(job_id)
        if status.status == "Completed":
            break
        time.sleep(0.05)
    
    status = system_tagger.status(job_id)
    assert status.status == "Completed"
    
    # GPU should not have been allocated
    job = system_tagger.jobs[job_id]
    assert len(job.gpus_used) == 0


def test_status_returns_copy(system_tagger):
    """Test that status method returns a copy, not the original object"""
    container = MockTagContainer()
    resources = {"A6000": 1}
    
    job_id = system_tagger.start(container, resources)
    
    status1 = system_tagger.status(job_id)
    status2 = system_tagger.status(job_id)
    
    # Should be equal but not the same object
    assert status1.status == status2.status
    assert status1 is not status2


def test_stop_queued_job_doesnt_free_resources(system_tagger):
    """Test that stopping a queued job doesn't incorrectly free resources"""
    # Start enough jobs to fill all GPUs and queue some
    containers = [MockTagContainer(work_duration=1.0) for _ in range(4)]  # Long duration
    resources = {"A6000": 1}  # Each needs 1 GPU, but we only have 2

    total_resources = system_tagger.active_resources.copy()
    
    job_ids = []
    for container in containers:
        job_id = system_tagger.start(container, resources)
        job_ids.append(job_id)
    
    # Let the first jobs start and others queue
    time.sleep(0.1)
    
    # Find a queued job
    queued_job_id = None
    for job_id in job_ids:
        status = system_tagger.status(job_id)
        if status.status == "Queued":
            queued_job_id = job_id
            break
    
    assert queued_job_id is not None, "Should have at least one queued job"
    
    # Record resources before stopping the queued job
    resources_before = system_tagger.active_resources.copy()

    assert resources_before["A6000"] == 0
    
    # Stop the queued job
    system_tagger.stop(queued_job_id)
    
    # Resources should be the same (no resources were reserved for queued job)
    resources_after = system_tagger.active_resources
    assert resources_after == resources_before, f"Resources changed from {resources_before} to {resources_after} when stopping queued job"
    
    # The job should be stopped
    status = system_tagger.status(queued_job_id)
    assert status.status == "Stopped"
    
    # Clean up - stop all jobs
    for job_id in job_ids:
        if job_id != queued_job_id:
            system_tagger.stop(job_id)

    # check that resources are fully restored
    final_resources = system_tagger.active_resources
    assert final_resources == total_resources, f"Final resources {final_resources} do not match total {total_resources}"