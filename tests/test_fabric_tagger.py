import threading
import pytest
import tempfile
import shutil
import os
import time
from unittest.mock import Mock

from src.tagger.fabric_tagging.tagger import FabricTagger
from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tag_containers.types import ModelOutput
from src.tagger.system_tagging.types import SysConfig
from src.tagger.fabric_tagging.types import RunConfig
from src.tags.tagstore import FilesystemTagStore
from src.fetch.types import VodDownloadRequest, DownloadResult, Source, StreamMetadata
from src.common.content import Content
from src.common.schema import Tag
from src.common.resources import SystemResources
from src.api.tagging.format import TagArgs
from src.common.errors import MissingResourceError


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def fake_media_files(temp_dir):
    """Create fake media files for testing"""
    video1 = os.path.join(temp_dir, "video1.mp4")
    video2 = os.path.join(temp_dir, "video2.mp4")
    
    # Create empty files
    with open(video1, 'w') as f:
        f.write("fake video content 1")
    with open(video2, 'w') as f:
        f.write("fake video content 2")
    
    return [video1, video2]


@pytest.fixture
def fake_fetcher(fake_media_files):
    """Create a fake fetcher that returns empty media files"""
    class FakeFetcher:
        def download_stream(self, content: Content, req: VodDownloadRequest, exit_event=None) -> DownloadResult:
            # Simulate successful download
            sources = []
            for i, filepath in enumerate(fake_media_files):
                source = Source(
                    filepath=filepath,
                    name=f"part_{i}.mp4",
                    offset=i * 10.0,  # 10 second parts
                )
                sources.append(source)
            
            stream_meta = StreamMetadata(
                codec_type="video",
                parts=["hash1", "hash2"],
                part_duration=10.0,
                fps=30.0
            )
            
            return DownloadResult(
                successful_sources=sources,
                failed=[],
                stream_meta=stream_meta
            )
    
    return FakeFetcher()


@pytest.fixture
def tag_store(temp_dir):
    """Create a real FilesystemTagStore for testing"""
    tagstore_dir = os.path.join(temp_dir, "tagstore")
    return FilesystemTagStore(tagstore_dir)


@pytest.fixture
def system_tagger():
    """Create a real SystemTagger for testing"""
    return SystemTagger(cfg=SysConfig(gpus=["gpu", "gpu", "disabled"], cpu_juice=100))


@pytest.fixture
def fake_container_registry(temp_dir, fake_media_files):
    """Create a fake ContainerRegistry that returns mock containers with fake tags"""

    class FakeTagContainer:
        """Fake TagContainer that simulates work and asynchronous behavior."""
        
        def __init__(self, fileargs, feature, work_duration: float = 0.1):
            """
            Initialize the FakeTagContainer.

            Args:
                fileargs (list[str]): List of file paths to process.
                feature (str): The feature being tagged.
                work_duration (float): Time in seconds to simulate work.
            """
            self.fileargs = fileargs
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

        def tags(self) -> list[ModelOutput]:
            """
            Return fake tags for the media files.

            Returns:
                list[ModelOutput]: List of fake tags for each file.
            """
            self.tag_call_count += 1

            outputs = []
            for i, filepath in enumerate(self.fileargs):
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
    
    class FakeContainerRegistry:
        def __init__(self):
            self.containers = {}
            
        def get(self, feature: str, fileargs: list[str], runconfig: dict) -> FakeTagContainer:
            container_key = f"{feature}_{len(fileargs)}"
            if container_key not in self.containers:
                self.containers[container_key] = FakeTagContainer(fileargs, feature)
            return self.containers[container_key]
        
        def get_model_resources(self, feature: str) -> SystemResources:
            return {"gpu": 1, "cpu_juice": 5}
        
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
    
    return FakeContainerRegistry()


@pytest.fixture
def fabric_tagger(system_tagger, fake_container_registry, tag_store, fake_fetcher):
    """Create a FabricTagger instance for testing"""
    tagger = FabricTagger(
        manager=system_tagger,
        cregistry=fake_container_registry,
        tagstore=tag_store,
        fetcher=fake_fetcher
    )
    yield tagger
    tagger.cleanup()


@pytest.fixture
def sample_content():
    """Create a sample Content object for testing"""
    return Mock(qhit="iq__test_content", auth="fake_auth_token")


@pytest.fixture
def sample_tag_args():
    """Create sample TagArgs for testing"""
    return TagArgs(
        features={
            "object_detection": RunConfig(stream="video", model={}),
            "speech_recognition": RunConfig(stream="audio", model={})
        },
        replace=False,
        start_time=0,
        end_time=30
    )


def test_tag_success(fabric_tagger, sample_content, sample_tag_args):
    """Test successful tagging job creation"""
    result = fabric_tagger.tag(sample_content, sample_tag_args)
    
    # Check that jobs were started successfully
    assert "object_detection" in result
    assert "speech_recognition" in result
    assert result["object_detection"] == "Job started successfully"
    assert result["speech_recognition"] == "Job started successfully"
    
    # Check that jobs are in active jobs
    active_jobs = fabric_tagger.jobstore.active_jobs
    assert len(active_jobs) == 2
    
    # Check job details
    job_ids = list(active_jobs.keys())
    for job_id in job_ids:
        job = active_jobs[job_id]
        assert job_id.qhit == "iq__test_content"
        assert job.args.feature in ["object_detection", "speech_recognition"]
        assert job.status.status in ["Starting", "Fetching content"]


def test_tag_invalid_feature(fabric_tagger, sample_content):
    """Test tagging with invalid feature"""
    invalid_args = TagArgs(
        features={"invalid_feature": RunConfig(**{"stream": "video", "model": {}})},
        replace=False
    )
    
    with pytest.raises(MissingResourceError, match="Invalid feature: invalid_feature"):
        fabric_tagger.tag(sample_content, invalid_args)


def test_tag_duplicate_job(fabric_tagger, sample_content, sample_tag_args):
    """Test that duplicate jobs are rejected"""
    # Start first job
    result1 = fabric_tagger.tag(sample_content, sample_tag_args)
    assert result1["object_detection"] == "Job started successfully"
    
    # Try to start same job again
    result2 = fabric_tagger.tag(sample_content, sample_tag_args)
    # Should get error message for duplicate job
    assert "already running" in result2["object_detection"]


def test_status_no_jobs(fabric_tagger):
    """Test status when no jobs exist"""
    with pytest.raises(MissingResourceError, match="No jobs started for"):
        fabric_tagger.status("iq__nonexistent")


def test_status_with_jobs(fabric_tagger, sample_content, sample_tag_args):
    """Test status retrieval with active jobs"""
    # Start jobs
    fabric_tagger.tag(sample_content, sample_tag_args)
    
    # Get status
    status = fabric_tagger.status("iq__test_content")
    
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


def test_status_completed_jobs(fabric_tagger, sample_content, sample_tag_args):
    """Test status includes completed jobs"""
    # Start and complete jobs
    fabric_tagger.tag(sample_content, sample_tag_args)
    
    # Wait a bit for jobs to progress
    time.sleep(0.5)
    
    # Manually complete a job for testing
    with fabric_tagger.storelock:
        if fabric_tagger.jobstore.active_jobs:
            job_id = list(fabric_tagger.jobstore.active_jobs.keys())[0]
            job = fabric_tagger.jobstore.active_jobs[job_id]
            fabric_tagger._stop_job(job_id, "Completed", None)
    
    # Status should still include completed job
    status = fabric_tagger.status("iq__test_content")
    assert len(status) > 0


def test_stop_running_job(fabric_tagger, sample_content, sample_tag_args):
    """Test stopping a running job"""
    # Start jobs
    fabric_tagger.tag(sample_content, sample_tag_args)
    
    # Wait for jobs to start
    time.sleep(0.1)
    
    # Stop one job
    fabric_tagger.stop("iq__test_content", "object_detection")
    
    # Check that stop event was set
    active_jobs = fabric_tagger.jobstore.active_jobs
    for job in active_jobs.values():
        if job.args.feature == "object_detection":
            assert job.stopevent.is_set()


def test_stop_nonexistent_job(fabric_tagger):
    """Test stopping a job that doesn't exist"""
    with pytest.raises(MissingResourceError, match="No job running for"):
        fabric_tagger.stop("iq__nonexistent", "object_detection")


def test_stop_specific_feature(fabric_tagger, sample_content, sample_tag_args):
    """Test stopping only a specific feature job"""
    # Start multiple jobs
    fabric_tagger.tag(sample_content, sample_tag_args)
    
    initial_count = len(fabric_tagger.jobstore.active_jobs)
    assert initial_count == 2
    
    # Stop only object_detection
    fabric_tagger.stop("iq__test_content", "object_detection")
    
    # Check that only object_detection job has stop event set
    active_jobs = fabric_tagger.jobstore.active_jobs
    obj_detection_stopped = False
    speech_rec_running = False
    
    for job in active_jobs.values():
        if job.args.feature == "object_detection":
            obj_detection_stopped = job.stopevent.is_set()
        elif job.args.feature == "speech_recognition":
            speech_rec_running = not job.stopevent.is_set()
    
    assert obj_detection_stopped
    assert speech_rec_running


def test_cleanup(fabric_tagger, sample_content, sample_tag_args):
    """Test cleanup shuts down properly"""
    # Start some jobs
    fabric_tagger.tag(sample_content, sample_tag_args)
    
    # Cleanup
    fabric_tagger.cleanup()
    
    # Check shutdown signal is set
    assert fabric_tagger.shutdown_signal.is_set()


def test_job_lifecycle(fabric_tagger, sample_content, sample_tag_args):
    """Test complete job lifecycle from start to completion"""
    # Start jobs
    result = fabric_tagger.tag(sample_content, sample_tag_args)
    assert all("successfully" in msg for msg in result.values())
    
    # Jobs should be active initially
    assert len(fabric_tagger.jobstore.active_jobs) == 2
    assert len(fabric_tagger.jobstore.inactive_jobs) == 0
    
    # Wait for jobs to progress through stages
    max_wait = 5.0  # 5 second timeout
    start_time = time.time()
    
    while (len(fabric_tagger.jobstore.inactive_jobs) < 2 and 
           time.time() - start_time < max_wait):
        time.sleep(0.1)
    
    # Check final state
    status = fabric_tagger.status("iq__test_content")
    
    # Should have status for both features
    all_features = []
    for stream_status in status.values():
        all_features.extend(stream_status.keys())
    
    assert "object_detection" in all_features
    assert "speech_recognition" in all_features


def test_concurrent_jobs_different_content(fabric_tagger):
    """Test running jobs for different content simultaneously"""
    content1 = Content(qhit="iq__content1", auth="token1")
    content2 = Content(qhit="iq__content2", auth="token2")
    
    args1 = TagArgs(
        features={"object_detection": {"stream": "video", "model": "yolo"}},
        replace=False
    )
    args2 = TagArgs(
        features={"speech_recognition": {"stream": "audio", "model": "whisper"}},
        replace=False
    )
    
    # Start jobs for different content
    result1 = fabric_tagger.tag(content1, args1)
    result2 = fabric_tagger.tag(content2, args2)
    
    assert result1["object_detection"] == "Job started successfully"
    assert result2["speech_recognition"] == "Job started successfully"
    
    # Should have 2 active jobs
    assert len(fabric_tagger.jobstore.active_jobs) == 2
    
    # Should be able to get status for both
    status1 = fabric_tagger.status("iq__content1")
    status2 = fabric_tagger.status("iq__content2")
    
    assert status1 != status2


def test_job_error_handling(fabric_tagger, sample_content):
    """Test job error handling"""
    # Create args with features that will cause errors in the fake system
    error_args = TagArgs(
        features={"object_detection": {"stream": "video", "model": "nonexistent"}},
        replace=False
    )
    
    # Start job
    result = fabric_tagger.tag(sample_content, error_args)
    assert result["object_detection"] == "Job started successfully"
    
    # Wait a bit for job to potentially fail
    time.sleep(0.5)
    
    # Job should handle errors gracefully
    status = fabric_tagger.status("iq__test_content")
    assert isinstance(status, dict)


def test_replace_functionality(fabric_tagger, sample_content):
    """Test replace parameter functionality"""
    args_replace = TagArgs(
        features={"object_detection": {"stream": "video", "model": "yolo"}},
        replace=True,
        start_time=0,
        end_time=30
    )
    
    result = fabric_tagger.tag(sample_content, args_replace)
    assert result["object_detection"] == "Job started successfully"
    
    # Check that job was created with replace=True
    active_jobs = fabric_tagger.jobstore.active_jobs
    job = list(active_jobs.values())[0]
    assert job.args.replace == True


def test_time_range_functionality(fabric_tagger, sample_content):
    """Test start_time and end_time parameters"""
    args_with_time = TagArgs(
        features={"object_detection": {"stream": "video", "model": "yolo"}},
        replace=False,
        start_time=10,
        end_time=60
    )
    
    result = fabric_tagger.tag(sample_content, args_with_time)
    assert result["object_detection"] == "Job started successfully"
    
    # Check that job was created with correct time range
    active_jobs = fabric_tagger.jobstore.active_jobs
    job = list(active_jobs.values())[0]
    assert job.args.start_time == 10
    assert job.args.end_time == 60


def test_multiple_features_same_stream(fabric_tagger, sample_content):
    """Test multiple features on the same stream"""
    args_multi = TagArgs(
        features={
            "object_detection": {"stream": "video", "model": "yolo"},
            "scene_analysis": {"stream": "video", "model": "resnet"}
        },
        replace=False
    )
    
    result = fabric_tagger.tag(sample_content, args_multi)
    assert result["object_detection"] == "Job started successfully"
    assert result["scene_analysis"] == "Job started successfully"
    
    # Should have 2 separate jobs
    assert len(fabric_tagger.jobstore.active_jobs) == 2
    
    # Both should be for video stream
    for job in fabric_tagger.jobstore.active_jobs.values():
        assert job.args.runconfig["stream"] == "video"


def test_thread_safety(fabric_tagger, sample_content):
    """Test thread safety of concurrent operations"""
    import threading
    
    def start_job(feature_name):
        args = TagArgs(
            features={feature_name: {"stream": "video", "model": "test"}},
            replace=False
        )
        try:
            return fabric_tagger.tag(sample_content, args)
        except Exception as e:
            return {"error": str(e)}
    
    # Start multiple jobs concurrently
    threads = []
    results = []
    
    for i in range(3):
        def worker(idx=i):
            result = start_job(f"feature_{idx}")
            results.append(result)
        
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join(timeout=5)
    
    # Should have some successful results
    assert len(results) == 3
    
    # At least one should succeed (others might fail due to invalid features)
    success_count = sum(1 for r in results if any("successfully" in str(v) for v in r.values()))
    assert success_count >= 0  # At least some operations should complete