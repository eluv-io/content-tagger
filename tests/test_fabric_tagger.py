import threading
import pytest
import tempfile
import shutil
import os
import time
from unittest.mock import Mock

from src.tagger.fabric_tagging.tagger import FabricTagger
from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tag_containers.types import ModelConfig, ModelOutput
from src.tagger.system_tagging.types import SysConfig
from src.tagger.fabric_tagging.types import RunConfig, TagArgs
from src.tags.tagstore.tagstore import FilesystemTagStore
from src.fetch.types import DownloadRequest, DownloadResult, Source, StreamMetadata, VideoScope
from src.common.content import Content
from src.tags.tagstore.types import Tag, TagStoreConfig
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
        def __init__(self, timeout=0.1):
            self.timeout = timeout

        def download(self, content: Content, req: DownloadRequest, exit_event=None) -> DownloadResult:
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

            time.sleep(self.timeout)
            
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
    return FilesystemTagStore(TagStoreConfig(base_path=tagstore_dir))


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
    
    class FakeContainerRegistry:
        def __init__(self):
            self.containers = {}
            
        def get(self, feature: str, fileargs: list[str], runconfig: dict) -> FakeTagContainer:
            container_key = f"{feature}_{len(fileargs)}"
            if container_key not in self.containers:
                self.containers[container_key] = FakeTagContainer(fileargs, feature)
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
    
    return FakeContainerRegistry()


@pytest.fixture
def fabric_tagger(system_tagger, fake_container_registry, tag_store, fake_fetcher):
    """Create a FabricTagger instance for testing"""
    tagger = FabricTagger(
        system_tagger=system_tagger,
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
        scope=VideoScope(start_time=0, end_time=30)
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
        assert job.state.status.status in ["Starting", "Fetching content"]


def test_tag_invalid_feature(fabric_tagger, sample_content):
    """Test tagging with invalid feature"""
    invalid_args = TagArgs(
        features={"invalid_feature": RunConfig(**{"stream": "video", "model": {}})},
        replace=False,
        scope=VideoScope(start_time=None, end_time=None)
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
    time.sleep(1)
    
    # Check that job is done according to status
    status = fabric_tagger.status("iq__test_content")
    assert status["video"]["object_detection"]["status"] == "Completed"
    assert status["audio"]["speech_recognition"]["status"] == "Completed"


def test_stop_running_job(fabric_tagger, sample_content, sample_tag_args):
    """Test stopping a running job"""
    # Start jobs
    fabric_tagger.tag(sample_content, sample_tag_args)
    
    # Stop one job
    fabric_tagger.stop("iq__test_content", "object_detection")
    
    # Check that stop event was set
    status = fabric_tagger.status("iq__test_content")
    assert status["video"]["object_detection"]["status"] == "Stopped"
    assert status["audio"]["speech_recognition"]["status"] != "Stopped"

def test_stop_nonexistent_job(fabric_tagger):
    """Test stopping a job that doesn't exist"""
    with pytest.raises(MissingResourceError, match="No job running for"):
        fabric_tagger.stop("iq__nonexistent", "object_detection")

def test_stop_finished_job(fabric_tagger, sample_content, sample_tag_args):
    fabric_tagger.tag(sample_content, sample_tag_args)
    time.sleep(1)
    with pytest.raises(MissingResourceError, match="No job running for"):
        fabric_tagger.stop("iq__test_content", "object_detection")
    # check that both are Completed
    status = fabric_tagger.status("iq__test_content")
    assert status["video"]["object_detection"]["status"] == "Completed"
    assert status["audio"]["speech_recognition"]["status"] == "Completed"

def test_cleanup(fabric_tagger, sample_content, sample_tag_args):
    """Test cleanup shuts down properly"""
    # Start some jobs
    fabric_tagger.tag(sample_content, sample_tag_args)

    # Cleanup
    fabric_tagger.cleanup()

    time.sleep(1)
    
    # Check shutdown signal is set
    assert fabric_tagger.shutdown_signal.is_set()

    # check is_running for all containers
    status = fabric_tagger.status("iq__test_content")
    for stream, features in status.items():
        for feature, job_status in features.items():
            assert job_status["status"] == "Stopped"

    assert len(fabric_tagger.jobstore.active_jobs) == 0
    assert len(fabric_tagger.jobstore.inactive_jobs) == 2

    assert fabric_tagger.system_tagger.exit.is_set()

    assert len(fabric_tagger.system_tagger.q) == 0

    for job in fabric_tagger.system_tagger.jobs.values():
        assert job.stopevent.is_set()
        assert job.finished.is_set()
        assert job.container.is_running() is False

def test_many_concurrent_jobs(fabric_tagger):
    """Run many jobs and make sure that they all run"""
    contents = []
    for i in range(5):
        contents.append(Mock(qhit=f"iq__content{i}", auth=f"token{i}"))

    args = TagArgs(
        features={"object_detection": RunConfig(**{"stream": "video", "model": {}}),
                  "speech_recognition": RunConfig(**{"stream": "audio", "model": {}})},
        replace=False,
        scope=VideoScope(start_time=None, end_time=None)
    )

    results = []
    for content in contents:
        result = fabric_tagger.tag(content, args)
        results.append(result)

    for i, result in enumerate(results):
        assert result["object_detection"] == "Job started successfully"

    # Should have 5 active jobs
    assert len(fabric_tagger.jobstore.active_jobs) == 10

    # Should be able to get status for all
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

def test_tags_uploaded_during_and_after_job(
    fabric_tagger, 
    sample_content, 
    sample_tag_args
):
    # Start jobs
    status = fabric_tagger.tag(sample_content, sample_tag_args)
    assert "object_detection" in status
    assert "speech_recognition" in status
    assert all("successfully" in msg.lower() for msg in status.values())
    
    tag_counts = set()
    job_statuses = set()

    timeout = 3
    start = time.time()
    end = False
    while time.time() - start < timeout:
        st = fabric_tagger.status(sample_content.qhit)
        end = True
        for stream in st:
            for feature in st[stream]:
                tag_count = fabric_tagger.tagstore.count_tags(track=feature, stream=stream)
                status = st[stream][feature]
                if status["status"] != "Completed":
                    end = False
                job_statuses.add(status["status"])
                if status["status"] == "Fetching Content":
                    assert tag_count == 0
                if tag_count == 2:
                    assert status["tagging_progress"] == '50%'
                if status["status"] == "Completed":
                    assert status["tagging_progress"] == '100%'
                    assert tag_count == 4
                tag_counts.add(tag_count)
        if end:
            break
        time.sleep(0.01)

    assert end

    assert len(job_statuses) == 3
    assert 'Fetching content' in job_statuses
    assert 'Tagging content' in job_statuses
    assert 'Completed' in job_statuses

    assert len(tag_counts) == 3
    assert 0 in tag_counts
    assert 2 in tag_counts
    assert 4 in tag_counts