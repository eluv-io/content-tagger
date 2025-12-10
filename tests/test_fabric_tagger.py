import threading
import pytest
import os
import time
from unittest.mock import Mock, patch

from src.tags.tagstore.model import Track
from src.tags.tagstore.rest_tagstore import RestTagstore
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.tag_containers.model import MediaInput, ModelConfig, ModelTag
from src.tagging.scheduling.model import SysConfig
from src.tagging.fabric_tagging.model import TagArgs
from src.tag_containers.model import ContainerRequest
from src.fetch.model import *
from src.common.content import Content
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

    def tags(self) -> list[ModelTag]:
        """
        Return fake tags for the media files.

        Returns:
            list[ModelTag]: List of fake tags for each file.
        """
        self.tag_call_count += 1

        tags = []
        if not self.is_stopped:
            finished_files = self.fileargs[:-1]
        else:
            finished_files = self.fileargs

        for i, filepath in enumerate(finished_files):
            # Create fake tags based on the feature
            fake_tags = [
                ModelTag(
                    start_time=0,
                    end_time=5000,  # 5 seconds in ms
                    text=f"{self.feature}_tag_{i}",
                    frame_tags={},
                    source_media=filepath,
                    track=""
                ),
                ModelTag(
                    start_time=5000,
                    end_time=10000,  # 5-10 seconds in ms
                    text=f"{self.feature}_tag_{i}_2",
                    frame_tags={},
                    source_media=filepath,
                    track=""
                )
            ]

            tags.extend(fake_tags)

        return tags
    
    def name(self) -> str:
        return f"FakeContainer-{self.feature}"
    
    def required_resources(self):
        return {}

class PartialFailContainer(FakeTagContainer):
    """
    Always fails last tag
    """

    def tags(self) -> list[ModelTag]:
        tags = super().tags()
        sources = [t.source_media for t in tags]
        if len(sources) > 1:
            return [t for t in tags if t.source_media != sources[-1]]

        return tags
    
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
        return ["caption", "asr"]
    
    @property
    def cfg(self):
        # Mock config with modconfigs
        mock_cfg = Mock()
        mock_cfg.modconfigs = {
            "caption": Mock(type="video"),
            "asr": Mock(type="audio"),
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
                offset=i * 10000,  # 10 second parts
                wall_clock=time.time_ns() // 1_000_000  # current time in ms
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
            feature="caption",
            run_config={},
            scope=VideoScope(stream="video", start_time=0, end_time=30),
            replace=False,
            destination_qid=""
        ),
        TagArgs(
            feature="asr", 
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
    assert "caption" in features
    assert "asr" in features


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
    
    assert "caption" in all_features
    assert "asr" in all_features
    
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
    assert status["video"]["caption"]["status"] == "Completed"
    assert status["audio"]["asr"]["status"] == "Completed"


def test_stop_running_job(fabric_tagger, q, sample_tag_args):
    """Test stopping a running job"""
    # Start jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    # Stop one job
    fabric_tagger.stop(q.qhit, "caption", None)
    
    # Check that stop event was set
    status = fabric_tagger.status(q.qhit)
    assert status["video"]["caption"]["status"] == "Stopped"
    assert status["audio"]["asr"]["status"] != "Stopped"


def test_stop_nonexistent_job(fabric_tagger):
    """Test stopping a job that doesn't exist"""
    if isinstance(fabric_tagger.tagstore, RestTagstore):
        pytest.skip("Skipping test for RestTagstore cause I don't want to have to configure many live test objects")

    with pytest.raises(MissingResourceError):
        fabric_tagger.stop("iq__nonexistent", "caption", None)


def test_stop_finished_job(fabric_tagger, q, sample_tag_args):
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    time.sleep(1)
    with pytest.raises(MissingResourceError):
        fabric_tagger.stop(q.qhit, "caption", None)
    # check that both are Completed
    status = fabric_tagger.status(q.qhit)
    assert status["video"]["caption"]["status"] == "Completed"
    assert status["audio"]["asr"]["status"] == "Completed"


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
        contents.append(Mock(qid=f"iq__content{i}", qhit=f"iq__content{i}", auth=f"token{i}"))

    # Create individual TagArgs for each feature and content
    all_args = []
    for content in contents:
        all_args.extend([
            TagArgs(
                feature="caption",
                run_config={},
                scope=VideoScope(stream="video", start_time=0, end_time=30),
                replace=False,
                destination_qid=""
            ),
            TagArgs(
                feature="asr", 
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
        assert status["video"]["caption"]["status"] in ["Starting", "Fetching content"]
        assert status["audio"]["asr"]["status"] in ["Starting", "Fetching content"]

    time.sleep(2)
    for content in contents:
        status = fabric_tagger.status(content.qhit)
        assert status["video"]["caption"]["status"] == "Completed"
        assert status["audio"]["asr"]["status"] == "Completed"
        assert len(status["video"]["caption"]["failed"]) == 0
        assert len(status["audio"]["asr"]["failed"]) == 0


def test_tags_uploaded_during_and_after_job(
    fabric_tagger, 
    q, 
    sample_tag_args
):
    # Start jobs
    for args in sample_tag_args:
        result = fabric_tagger.tag(q, args)
        assert "successfully" in result.lower()

    track_mapping = fabric_tagger.cfg.uploader.model_params

    tracks = [t.default.name for t in track_mapping.values()]

    tag_counts = set()

    timeout = 3
    start = time.time()
    end = False
    while time.time() - start < timeout:
        end = True
        for track in tracks:
            tag_count = fabric_tagger.tagstore.count_tags(track=track, q=q)
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
        feature="caption",
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
        job_status = status["video"]["caption"]["status"]
        
        if job_status == "Failed":
            break
        elif job_status in ["Completed", "Stopped"]:
            pytest.fail(f"Expected job to fail, but got status: {job_status}")
        
        time.sleep(0.25)
    else:
        pytest.fail("Job did not fail within timeout period")
    
    # Verify final status is Failed
    final_status = fabric_tagger.status(q.qhit)
    assert final_status["video"]["caption"]["status"] == "Failed"
    
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
        feature="caption",
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
        job_status = status["video"]["caption"]["status"]
        
        if job_status == "Failed":
            break
        elif job_status in ["Completed", "Stopped"]:
            pytest.fail(f"Expected job to fail, but got status: {job_status}")
        
        time.sleep(0.25)
    else:
        pytest.fail("Job did not fail within timeout period")
    
    # Verify final status is Failed
    final_status = fabric_tagger.status(q.qhit)
    assert final_status["video"]["caption"]["status"] == "Failed"
    assert "message" in final_status["video"]["caption"]
    
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
        feature="caption",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=q.qid
    )
    
    # Start the job
    result = fabric_tagger.tag(q, args)
    assert "successfully" in result.lower()

    wait_tag(fabric_tagger, q.qhit, timeout=5)

    # check that only one was updated
    status = fabric_tagger.status(q.qhit)
    assert status["video"]["caption"]["status"] == "Completed"
    assert len(status["video"]["caption"]["failed"]) == 0
    assert status["video"]["caption"]["tagging_progress"] == "1/2"
    assert len(status["video"]["caption"]["missing_tags"]) == 1


def wait_tag(fabric_tagger, batch_id, timeout):
    start = time.time()
    while time.time() - start < timeout:
        status = fabric_tagger.status(batch_id)
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
        feature="caption",
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
        job_status = status["video"]["caption"]["status"]
        
        if job_status == "Failed":
            break
        elif job_status in ["Completed", "Stopped"]:
            pytest.fail(f"Expected job to fail due to exit code, but got: {job_status}")
        
        time.sleep(0.25)
    else:
        pytest.fail("Job did not fail within timeout period")
    
    # Verify final status is Failed
    final_status = fabric_tagger.status(q.qhit)
    assert final_status["video"]["caption"]["status"] == "Failed"
    
    # Verify job moved to inactive
    assert len(fabric_tagger.jobstore.active_jobs) == 0
    assert len(fabric_tagger.jobstore.inactive_jobs) == 1
    
    # Verify container stopped with exit code 1
    inactive_job = list(fabric_tagger.jobstore.inactive_jobs.values())[0]
    assert inactive_job.state.container.exit_code() == 1

def test_destination_qid_uploads_to_correct_qhit(sample_tag_args, fabric_tagger: FabricTagger, q, q2):
    """Test that when destination_qid is set, tags are uploaded to that qhit instead of source"""

    for args in sample_tag_args:
        args.destination_qid = q2.qid
        fabric_tagger.tag(q, args)

    # Wait for job to complete
    wait_tag(fabric_tagger, q.qhit, timeout=10)
    time.sleep(0.5)
    
    # Verify tags were uploaded to destination_qhit
    tag_count_destination = fabric_tagger.tagstore.count_tags(qhit=q2.qid, q=q2)
    assert tag_count_destination > 0

    # Verify no tags were uploaded to source qhit
    tag_count_source = fabric_tagger.tagstore.count_tags(qhit=q.qhit, q=q)
    assert tag_count_source == 0

def test_tags_have_timestamp_ms_field(fabric_tagger: FabricTagger, q: Content, sample_tag_args):
    """Test that uploaded tags include timestamp_ms in additional_info"""
    
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)

    wait_tag(fabric_tagger, q.qhit, timeout=5)
    
    tags = fabric_tagger.tagstore.find_tags(
        qhit=q.qhit,
        q=q
    )

    assert len(tags) > 0

    for tag in tags:
        assert "timestamp_ms" in tag.additional_info
        assert isinstance(tag.additional_info["timestamp_ms"], int)
        assert tag.additional_info["timestamp_ms"] > 0


def test_source_with_zero_tags_marked_as_missing(fabric_tagger, q):
    """Test that a source producing zero tags is marked as failed in job status"""
    
    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return PartialFailContainer(req.media_input, req.model_id)
    
    args = TagArgs(
        feature="caption",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=""
    )
    
    with patch.object(FakeContainerRegistry, 'get', side_effect=get_side_effect):
        fabric_tagger.tag(q, args)
        wait_tag(fabric_tagger, q.qhit, timeout=5)
    
    status = fabric_tagger.status(q.qhit)
    job_status = status["video"]["caption"]
    assert job_status["status"] == "Completed"
    assert len(job_status["missing_tags"]) == 1

def test_track_override_uploads_to_multiple_tracks(fabric_tagger, q):
    """Test that model tags with different tracks are uploaded to different tagstore tracks"""
    
    class MultiTrackContainer(FakeTagContainer):
        """Container that outputs tags with different tracks"""
        
        def tags(self) -> list[ModelTag]:
            tags = []
            finished_files = self.fileargs if self.is_stopped else self.fileargs[:-1]
            
            for i, filepath in enumerate(finished_files):
                # Create tags with default track (empty string)
                tags.append(ModelTag(
                    start_time=0,
                    end_time=5000,
                    text=f"default_track_tag_{i}",
                    frame_tags={},
                    source_media=filepath,
                    track=""
                ))
                
                # Create tags with "pretty" track
                tags.append(ModelTag(
                    start_time=5000,
                    end_time=10000,
                    text=f"pretty_track_tag_{i}",
                    frame_tags={},
                    source_media=filepath,
                    track="pretty"
                ))
            
            return tags
    
    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return MultiTrackContainer(req.media_input, req.model_id)
    
    args = TagArgs(
        feature="asr",
        run_config={},
        scope=VideoScope(stream="audio", start_time=0, end_time=30),
        replace=False,
        destination_qid=""
    )
    
    with patch.object(FakeContainerRegistry, 'get', side_effect=get_side_effect):
        fabric_tagger.tag(q, args)
        wait_tag(fabric_tagger, q.qhit, timeout=5)
    
    # Verify tags were uploaded to both tracks
    default_track = "speech_to_text"
    override_track = "auto_captions"
    
    default_tags = fabric_tagger.tagstore.find_tags(
        q=q,
        track=default_track
    )
    
    override_tags = fabric_tagger.tagstore.find_tags(
        q=q,
        track=override_track
    )
    
    # Should have 2 tags per source (2 sources total = 4 tags per track)
    assert len(default_tags) == 2, f"Expected 2 tags on default track, got {len(default_tags)}"
    assert len(override_tags) == 2, f"Expected 2 tags on override track, got {len(override_tags)}"

def test_uploaded_track_label(fabric_tagger, q):
    """Test that uploaded tags have correct track labels based on model params"""
    
    args = TagArgs(
        feature="caption",
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=30),
        replace=False,
        destination_qid=""
    )
    
    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qhit, timeout=5)
    
    track_arg = fabric_tagger.cfg.uploader.model_params["caption"].default
    
    track = fabric_tagger.tagstore.get_track(
        qhit=q.qhit,
        q=q,
        name=track_arg.name
    )
    
    assert isinstance(track, Track)
    assert track.name == track_arg.name
    assert track.label == track_arg.label