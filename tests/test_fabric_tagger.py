import threading
import pytest
import os
import time
from unittest.mock import Mock, patch
from collections import defaultdict

from src.tags.tagstore.rest_tagstore import RestTagstore
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.tag_containers.model import MediaInput, ModelConfig, ModelOutput
from src.tagging.scheduling.model import SysConfig
from src.tagging.fabric_tagging.model import TagArgs
from src.tag_containers.model import ContainerRequest
from src.fetch.model import *
from src.common.content import Content
from src.tags.tagstore.types import Tag
from src.common.errors import MissingResourceError

class FakeTagContainer:
    """Fake TagContainer that simulates work and asynchronous behavior."""
    
    def __init__(self, media: MediaInput, feature, work_duration: float = 0.25):
        """
        Initialize the FakeTagContainer.
    
        Args:
            media: List of file paths or directory to process.
            feature (str): The feature being tagged.
            work_duration (float): Time in seconds to simulate work.
        """
        if isinstance(media, str):
            self.fileargs = [os.path.join(media, f) for f in os.listdir(media)]
        else:
            self.fileargs = media
        self.feature = feature
        self.work_duration = work_duration
        self.is_started = False
        self.is_stopped = False
        self.container = Mock()
        self.container.attrs = {"State": {"ExitCode": 0}}
        self.tag_call_count = 0
        self.worker_thread = None

    def start(self, gpu_idx: int | None = None) -> None:
        """
        Start the container and simulate work in a background thread.

        Args:
            gpu_idx (int | None): GPU index to use (if applicable).
        """
        
        self.gpu_idx = gpu_idx

        # Simulate work in a background thread
        def work():
            self.is_started = True
            time.sleep(self.work_duration)
            self.is_stopped = True

        self.worker_thread = threading.Thread(target=work, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        """
        Stop the container and terminate the work.
        """
        self.is_stopped = True

    def is_running(self) -> bool:
        """
        Check if the container is still running.

        Returns:
            bool: True if the container is running, False otherwise.
        """
        return self.is_started and not self.is_stopped
    
    def exit_code(self) -> int | None:
        """
        Get the exit code of the container.

        Returns:
            int | None: Exit code if available, None otherwise.
        """
        if self.is_stopped:
            return 0
        return None
    
    def send_eof(self) -> None:
        pass

    def tags(self) -> list[ModelOutput]:
        """
        Return fake tags for the media files.

        Returns:
            list[ModelOutput]: List of fake tags for each file.
        """
        self.tag_call_count += 1

        outputs = []
        if not self.is_stopped:
            finished_files = self.fileargs[:-1]
        else:
            finished_files = self.fileargs

        for i, filepath in enumerate(finished_files):
            # Create fake tags based on the feature
            fake_tags = [
                Tag(
                    start_time=0,
                    end_time=5000,  # 5 seconds in ms
                    text=f"{self.feature}_tag_{i}",
                    additional_info={"confidence": 0.9},
                    source="",
                    jobid=""
                ),
                Tag(
                    start_time=5000,
                    end_time=10000,  # 5-10 seconds in ms
                    text=f"{self.feature}_tag_{i}_2",
                    additional_info={"confidence": 0.8},
                    source="",
                    jobid=""
                )
            ]

            output = ModelOutput(
                source_media=filepath,
                tags=fake_tags
            )
            outputs.append(output)

        return outputs
    
    def name(self) -> str:
        return f"FakeContainer-{self.feature}"
    
    def required_resources(self):
        return {}

class PartialFailContainer(FakeTagContainer):
    """
    Always fails last tag
    """

    def tags(self) -> list[ModelOutput]:
        outputs = super().tags()
        if len(outputs) > 1:
            outputs = outputs[:-1]

        return outputs
    
    def send_eof(self) -> None:
        pass

class FakeContainerRegistry:
    def __init__(self):
        self.containers = {}
        
    def get(self, req: ContainerRequest) -> FakeTagContainer:
        container_key = f"{req.model_id}_{req.media_input}"
        if container_key not in self.containers:
            self.containers[container_key] = FakeTagContainer(req.media_input, req.model_id)
        return self.containers[container_key]
    
    def get_model_config(self, feature: str) -> ModelConfig:
        return Mock(resources={"gpu": 1, "cpu_juice": 5})
    
    def services(self) -> list[str]:
        return ["object_detection", "speech_recognition", "scene_analysis"]
    
    @property
    def cfg(self):
        # Mock config with modconfigs
        mock_cfg = Mock()
        mock_cfg.modconfigs = {
            "object_detection": Mock(type="video"),
            "speech_recognition": Mock(type="audio"), 
            "scene_analysis": Mock(type="frame")
        }
        return mock_cfg

class FakeWorker(FetchSession):
    """Fake DownloadWorker that creates test files"""
    def __init__(self, output_dir: str, timeout: float = 0.1):
        self.output_dir = output_dir
        self.timeout = timeout
        self._metadata = VideoMetadata(
            parts=["hash1", "hash2"],
            part_duration=10.0,
            fps=30.0,
            codec_type="video"
        )

    def metadata(self) -> VideoMetadata:
        return self._metadata

    def download(self) -> DownloadResult:
        """Create fake media files for testing"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        video1 = os.path.join(self.output_dir, "video1.mp4")
        video2 = os.path.join(self.output_dir, "video2.mp4")
        
        # Create empty files
        with open(video1, 'w') as f:
            f.write("fake video content 1")
        with open(video2, 'w') as f:
            f.write("fake video content 2")

        # Simulate work
        time.sleep(self.timeout)

        # Simulate successful download
        sources = []
        for i, filepath in enumerate([video1, video2]):
            source = Source(
                filepath=filepath,
                name=f"part_{i}.mp4",
                offset=i * 10.0,  # 10 second parts
            )
            sources.append(source)
        
        return DownloadResult(
            sources=sources,
            failed=[],
            done=True
        )

    @property
    def path(self) -> str:
        return self.output_dir

@pytest.fixture
def fake_fetcher(media_dir):
    """Create a fake fetcher that returns FakeWorker"""
    class FakeFetcher:
        def __init__(self):
            self.config = Mock(author="tagger", max_downloads=2)

        def get_session(self, q: Content, req: DownloadRequest, exit_event=None):
            """Return a FakeWorker"""
            return FakeWorker(output_dir=req.output_dir, timeout=0.1)
    
    return FakeFetcher()


@pytest.fixture
def system_tagger():
    """Create a real ContainerScheduler for testing"""
    return ContainerScheduler(cfg=SysConfig(gpus=["gpu", "gpu", "disabled"], resources={"cpu_juice": 100}))


@pytest.fixture
def fake_container_registry():
    """Create a fake ContainerRegistry that returns mock containers with fake tags"""
    return FakeContainerRegistry()


@pytest.fixture
def fabric_tagger(system_tagger, fake_container_registry, tag_store, fake_fetcher, tagger_config):
    """Create a FabricTagger instance for testing"""
    tagger = FabricTagger(
        system_tagger=system_tagger,
        cregistry=fake_container_registry,
        tagstore=tag_store,
        fetcher=fake_fetcher,
        cfg=tagger_config
    )
    yield tagger
    if not tagger.shutdown_requested:
        tagger.cleanup()


@pytest.fixture
def sample_tag_args():
    """Create sample TagArgs for testing - now returns list of TagArgs"""
    return [
        TagArgs(
            feature="object_detection",
            run_config={},
            scope=VideoScope(stream="video", start_time=0, end_time=30),
            replace=False,
            destination_qid=""
        ),
        TagArgs(
            feature="speech_recognition", 
            run_config={},
            scope=VideoScope(stream="audio", start_time=0, end_time=30),
            replace=False,
            destination_qid=""
        )
    ]


def test_tag_success(fabric_tagger, q, sample_tag_args):
    """Test successful tagging job creation"""
    results = []
    for args in sample_tag_args:
        result = fabric_tagger.tag(q, args)
        results.append(result)
    
    # Check that jobs were started successfully
    assert all("successfully" in result.lower() for result in results)
    
    # Check that jobs are in active jobs
    active_jobs = fabric_tagger.jobstore.active_jobs
    assert len(active_jobs) == 2
    
    # Check job details
    job_ids = list(active_jobs.keys())
    features = [active_jobs[job_id].args.feature for job_id in job_ids]
    assert "object_detection" in features
    assert "speech_recognition" in features


def test_tag_invalid_feature(fabric_tagger, q):
    """Test tagging with invalid feature"""
    invalid_args = TagArgs(
        feature="invalid_feature",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=""
    )
    
    with pytest.raises(MissingResourceError, match="Invalid feature: invalid_feature"):
        fabric_tagger.tag(q, invalid_args)


def test_tag_duplicate_job(fabric_tagger, q, sample_tag_args):
    """Test that duplicate jobs are rejected"""
    # Start first job
    result1 = fabric_tagger.tag(q, sample_tag_args[0])
    assert "successfully" in result1.lower()
    
    # Try to start same job again
    result2 = fabric_tagger.tag(q, sample_tag_args[0])
    # Should get error message for duplicate job
    assert "already running" in result2


def test_status_no_jobs(fabric_tagger):
    """Test status when no jobs exist"""
    if isinstance(fabric_tagger.tagstore, RestTagstore):
        pytest.skip("Skipping test for RestTagstore cause I don't want to have to configure many live test objects")

    with pytest.raises(MissingResourceError, match="No jobs started for"):
        fabric_tagger.status("iq__nonexistent")


def test_status_with_jobs(fabric_tagger, q, sample_tag_args):
    """Test status retrieval with active jobs"""
    # Start jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    # Get status
    status = fabric_tagger.status(q.qhit)
    
    # Check structure
    assert isinstance(status, dict)
    assert "video" in status or "audio" in status
    
    # Check that we have status for our features
    all_features = []
    for stream_status in status.values():
        all_features.extend(stream_status.keys())
    
    assert "object_detection" in all_features
    assert "speech_recognition" in all_features
    
    # Check status format
    for stream, features in status.items():
        for feature, job_status in features.items():
            assert "status" in job_status
            assert "time_running" in job_status
            assert "tagging_progress" in job_status
            assert "failed" in job_status


def test_status_completed_jobs(fabric_tagger, q, sample_tag_args):
    """Test status includes completed jobs"""
    # Start and complete jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    # Wait a bit for jobs to progress
    time.sleep(1)
    
    # Check that job is done according to status
    status = fabric_tagger.status(q.qhit)
    assert status["video"]["object_detection"]["status"] == "Completed"
    assert status["audio"]["speech_recognition"]["status"] == "Completed"


def test_stop_running_job(fabric_tagger, q, sample_tag_args):
    """Test stopping a running job"""
    # Start jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    # Stop one job
    fabric_tagger.stop(q.qhit, "object_detection", None)
    
    # Check that stop event was set
    status = fabric_tagger.status(q.qhit)
    assert status["video"]["object_detection"]["status"] == "Stopped"
    assert status["audio"]["speech_recognition"]["status"] != "Stopped"


def test_stop_nonexistent_job(fabric_tagger):
    """Test stopping a job that doesn't exist"""
    if isinstance(fabric_tagger.tagstore, RestTagstore):
        pytest.skip("Skipping test for RestTagstore cause I don't want to have to configure many live test objects")

    with pytest.raises(MissingResourceError):
        fabric_tagger.stop("iq__nonexistent", "object_detection", None)


def test_stop_finished_job(fabric_tagger, q, sample_tag_args):
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    time.sleep(1)
    with pytest.raises(MissingResourceError):
        fabric_tagger.stop(q.qhit, "object_detection", None)
    # check that both are Completed
    status = fabric_tagger.status(q.qhit)
    assert status["video"]["object_detection"]["status"] == "Completed"
    assert status["audio"]["speech_recognition"]["status"] == "Completed"


def test_cleanup(fabric_tagger, q, sample_tag_args):
    """Test cleanup shuts down properly"""
    # Start some jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)

    time.sleep(1)

    # Cleanup
    fabric_tagger.cleanup()

    time.sleep(1)
    
    # Check shutdown signal is set
    assert fabric_tagger.shutdown_requested
    
    # check is_running for all containers
    assert len(fabric_tagger.jobstore.active_jobs) == 0
    assert len(fabric_tagger.jobstore.inactive_jobs) == 2

    assert fabric_tagger.system_tagger.exit_requested

    for job in fabric_tagger.system_tagger.jobs.values():
        assert job.finished.is_set()
        assert job.container.is_running() is False


def test_many_concurrent_jobs(fabric_tagger):
    if isinstance(fabric_tagger.tagstore, RestTagstore):
        pytest.skip("Skipping test for RestTagstore cause I don't want to have to configure many live test objects")

    """Run many jobs and make sure that they all run"""
    contents = []
    for i in range(5):
        contents.append(Mock(qhit=f"iq__content{i}", auth=f"token{i}"))

    # Create individual TagArgs for each feature and content
    all_args = []
    for content in contents:
        all_args.extend([
            TagArgs(
                feature="object_detection",
                run_config={},
                scope=VideoScope(stream="video", start_time=0, end_time=30),
                replace=False,
                destination_qid=""
            ),
            TagArgs(
                feature="speech_recognition", 
                run_config={},
                scope=VideoScope(stream="audio", start_time=0, end_time=30),
                replace=False,
                destination_qid=""
            )
        ])

    results = []
    content_idx = 0
    for i, args in enumerate(all_args):
        content = contents[content_idx]
        result = fabric_tagger.tag(content, args)
        results.append(result)
        if i % 2 == 1:  # Move to next content after every 2 jobs
            content_idx += 1

    for result in results:
        assert "successfully" in result.lower()

    assert len(fabric_tagger.jobstore.active_jobs) == 10

    statuses = []
    for content in contents:
        statuses.append(fabric_tagger.status(content.qhit))
    for status in statuses:
        assert status["video"]["object_detection"]["status"] in ["Starting", "Fetching content"]
        assert status["audio"]["speech_recognition"]["status"] in ["Starting", "Fetching content"]

    time.sleep(2)
    for content in contents:
        status = fabric_tagger.status(content.qhit)
        assert status["video"]["object_detection"]["status"] == "Completed"
        assert status["audio"]["speech_recognition"]["status"] == "Completed"
        assert len(status["video"]["object_detection"]["failed"]) == 0
        assert len(status["audio"]["speech_recognition"]["failed"]) == 0


def test_tags_uploaded_during_and_after_job(
    fabric_tagger, 
    q, 
    sample_tag_args
):
    # Start jobs
    for args in sample_tag_args:
        result = fabric_tagger.tag(q, args)
        assert "successfully" in result.lower()

    # Map streams to features
    streams_to_features = defaultdict(list)
    for args in sample_tag_args:
        streams_to_features[args.scope.stream].append(args.feature)

    tag_counts = set()

    timeout = 3
    start = time.time()
    end = False
    while time.time() - start < timeout:
        end = True
        for stream, features in streams_to_features.items():
            for feature in features:
                tag_count = fabric_tagger.tagstore.count_tags(track=feature, stream=stream, q=q)
                tag_counts.add(tag_count)
                if tag_count < 4:
                    end = False
        if end:
            break
        time.sleep(0.01)

    assert end

    assert len(tag_counts) == 3
    assert 0 in tag_counts
    assert 2 in tag_counts
    assert 4 in tag_counts


def test_tags_uploaded_during_and_after_job_through_status(
    fabric_tagger, 
    q, 
    sample_tag_args
):
    # Same as test_tags_uploaded_during_and_after_job but using status to check job status instead of querying tagstore.
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    timeout = 3
    start = time.time()
    end = False
    percentages = set()
    while time.time() - start < timeout:
        st = fabric_tagger.status(q.qhit)
        end = True
        for stream in st:
            for feature in st[stream]:
                status = st[stream][feature]
                percentages.add(status["tagging_progress"])
                if status["tagging_progress"] != '2/2':
                    end = False
        if end:
            break
        time.sleep(0.01)

    assert end

    assert len(percentages) == 4
    assert "0/2" in percentages
    assert "1/2" in percentages
    assert "2/2" in percentages


@patch.object(FakeTagContainer, 'tags')
def test_container_tags_method_fails(mock_tags, fabric_tagger, q):
    """Test that when container.tags() fails, the job fails gracefully and stops the container"""
    
    # Configure the mock to raise an exception
    mock_tags.side_effect = RuntimeError("Simulated container tags() failure")
    
    # Create args for a simple job
    args = TagArgs(
        feature="object_detection",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=q.qid
    )
    
    # Start the job
    result = fabric_tagger.tag(q, args)
    assert "successfully" in result.lower()
    
    # Wait for the job to fail
    timeout = 3
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = fabric_tagger.status(q.qhit)
        job_status = status["video"]["object_detection"]["status"]
        
        if job_status == "Failed":
            break
        elif job_status in ["Completed", "Stopped"]:
            pytest.fail(f"Expected job to fail, but got status: {job_status}")
        
        time.sleep(0.25)
    else:
        pytest.fail("Job did not fail within timeout period")
    
    # Verify final status is Failed
    final_status = fabric_tagger.status(q.qhit)
    assert final_status["video"]["object_detection"]["status"] == "Failed"
    
    # Verify the job is moved to inactive jobs
    assert len(fabric_tagger.jobstore.active_jobs) == 0
    assert len(fabric_tagger.jobstore.inactive_jobs) == 1
    
    # Verify the container was stopped
    inactive_job = list(fabric_tagger.jobstore.inactive_jobs.values())[0]
    time.sleep(0.25)  # Give a moment for the container to stop
    assert inactive_job.state.container.is_stopped == True
    assert inactive_job.stop_event.is_set()
    
    # Verify no tags were uploaded due to the failure
    tag_count = fabric_tagger.tagstore.count_tags(qhit=q.qhit, q=q)
    assert tag_count == 0
    
    # Verify the tags method was actually called
    assert mock_tags.call_count == 1

@patch.object(FabricTagger, '_start_new_container')
def test_start_new_container_fails(mock_process, fabric_tagger, q):
    """Test that when _start_new_container fails, job transitions to Failed state"""

    # Configure the mock to raise an exception
    mock_process.side_effect = RuntimeError("Simulated tagging phase processing failure")
    
    # Create args for a simple job
    args = TagArgs(
        feature="object_detection",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=q.qid
    )
    
    # Start the job
    fabric_tagger.tag(q, args)
    
    # Wait for the job to fail
    timeout = 2
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = fabric_tagger.status(q.qhit)
        job_status = status["video"]["object_detection"]["status"]
        
        if job_status == "Failed":
            break
        elif job_status in ["Completed", "Stopped"]:
            pytest.fail(f"Expected job to fail, but got status: {job_status}")
        
        time.sleep(0.25)
    else:
        pytest.fail("Job did not fail within timeout period")
    
    # Verify final status is Failed
    final_status = fabric_tagger.status(q.qhit)
    assert final_status["video"]["object_detection"]["status"] == "Failed"
    assert "message" in final_status["video"]["object_detection"]
    
    # Verify the job is moved to inactive jobs
    assert len(fabric_tagger.jobstore.active_jobs) == 0
    assert len(fabric_tagger.jobstore.inactive_jobs) == 1

    # Verify _start_new_container was actually called
    assert mock_process.call_count >= 1

@patch.object(FakeContainerRegistry, 'get')
def test_failed_tag(mock_get, fabric_tagger, q):
    # Configure the mock to return PartialFailContainer
    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return PartialFailContainer(req.media_input, req.model_id)
    
    mock_get.side_effect = get_side_effect
    
    # Create args for a simple job
    args = TagArgs(
        feature="object_detection",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=q.qid
    )
    
    # Start the job
    result = fabric_tagger.tag(q, args)
    assert "successfully" in result.lower()

    wait_tag(fabric_tagger, q.qhit, timeout=5)

    # check that one failed
    status = fabric_tagger.status(q.qhit)
    assert status["video"]["object_detection"]["status"] == "Completed"
    assert len(status["video"]["object_detection"]["failed"]) == 1


def wait_tag(fabric_tagger, jobid, timeout):
    start = time.time()
    while time.time() - start < timeout:
        status = fabric_tagger.status(jobid)
        for stream in status:
            for feature in status[stream]:
                if status[stream][feature]["status"] == "Completed":
                    return
                elif status[stream][feature]["status"] == "Failed":
                    pytest.fail(f"Job failed: {status[stream][feature]['error']}")
                time.sleep(0.25)
    pytest.fail("Job did not complete within timeout period")


@patch.object(FakeTagContainer, 'exit_code', autospec=True)
def test_container_nonzero_exit_code(mock_exit_code, fabric_tagger, q):
    """Test that a container with non-zero exit code causes job to fail"""
    
    # Configure mock to return container that fails
    def ec_side_effect(self):
        if self.is_stopped:
            return 1
        return None
    
    mock_exit_code.side_effect = ec_side_effect
    
    # Create args for a simple job
    args = TagArgs(
        feature="object_detection",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=q.qid
    )
    
    # Start the job
    result = fabric_tagger.tag(q, args)
    assert "successfully" in result.lower()
    
    # Wait for the job to fail due to non-zero exit code
    timeout = 3
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = fabric_tagger.status(q.qhit)
        job_status = status["video"]["object_detection"]["status"]
        
        if job_status == "Failed":
            break
        elif job_status in ["Completed", "Stopped"]:
            pytest.fail(f"Expected job to fail due to exit code, but got: {job_status}")
        
        time.sleep(0.25)
    else:
        pytest.fail("Job did not fail within timeout period")
    
    # Verify final status is Failed
    final_status = fabric_tagger.status(q.qhit)
    assert final_status["video"]["object_detection"]["status"] == "Failed"
    
    # Verify job moved to inactive
    assert len(fabric_tagger.jobstore.active_jobs) == 0
    assert len(fabric_tagger.jobstore.inactive_jobs) == 1
    
    # Verify container stopped with exit code 1
    inactive_job = list(fabric_tagger.jobstore.inactive_jobs.values())[0]
    assert inactive_job.state.container.exit_code() == 1

@patch.object(FakeTagContainer, 'tags')
def test_destination_qid_uploads_to_correct_qhit(mock_tags, fabric_tagger, q):
    """Test that when destination_qid is set, tags are uploaded to that qhit instead of source"""
    
    # Mock container.tags to return fake tags
    mock_tags.return_value = [
        ModelOutput(
            source_media="/fake/path/video1.mp4",
            tags=[
                Tag(
                    start_time=0,
                    end_time=5000,
                    text="test_tag",
                    additional_info={"confidence": 0.9},
                    source="",
                    jobid=""
                )
            ]
        )
    ]
    
    destination_qhit = "iq__destination"
    
    args = TagArgs(
        feature="object_detection",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=destination_qhit
    )

    fabric_tagger.tag(q, args)
    
    # Wait for job to complete
    wait_tag(fabric_tagger, q.qhit, timeout=5)
    
    # Verify tags were uploaded to destination_qhit
    tag_count_destination = fabric_tagger.tagstore.count_tags(qhit=destination_qhit, q=q)
    assert tag_count_destination == 1

    # Verify no tags were uploaded to source qhit
    tag_count_source = fabric_tagger.tagstore.count_tags(qhit=q.qhit, q=q)
    assert tag_count_source == 0