import os
import threading
import time
from unittest.mock import Mock
import pytest

from src.common.content import Content
from src.fetch.model import DownloadRequest, DownloadResult, FetchSession, Source, VideoMetadata, VideoScope
from src.fetch.model import VideoScope
from src.tag_containers.model import ContainerRequest, MediaInput, ModelConfig, ModelTag
from src.tagging.fabric_tagging.model import FabricTaggerConfig, TagArgs
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.tagging.scheduling.model import SysConfig
from src.tagging.uploading.config import ModelUploadArgs, TrackArgs, UploaderConfig


@pytest.fixture
def media_dir(temp_dir: str) -> str:
    """Create a media directory for testing"""
    media_path = os.path.join(temp_dir, "media")
    os.makedirs(media_path, exist_ok=True)
    return media_path

@pytest.fixture
def tagger_config(media_dir) -> FabricTaggerConfig:
    return FabricTaggerConfig(
        media_dir=media_dir, 
        uploader=UploaderConfig(
            model_params={
                "caption": ModelUploadArgs(
                    default=TrackArgs(name="object_detection", label="Object Detection")
                ),
                "asr": ModelUploadArgs(
                    default=TrackArgs(name="speech_to_text", label="Speech to Text"),
                    overrides={
                        "pretty": TrackArgs(name="auto_captions", label="Auto Captions")
                    }
                ),
            }
        ),
    )

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
def make_tag_args():
    """Factory for consistent TagArgs construction in tests."""
    def _make(
        feature: str = "caption",
        stream: str | None = None,
        destination_qid: str = "",
        replace: bool = False,
        run_config: dict | None = None,
        start_time: int = 0,
        end_time: int = 30,
        max_fetch_retries: int = 3
    ) -> TagArgs:
        if stream is None:
            stream = "audio" if feature == "asr" else "video"
        return TagArgs(
            feature=feature,
            run_config=run_config or {},
            scope=VideoScope(stream=stream, start_time=start_time, end_time=end_time),
            replace=replace,
            destination_qid=destination_qid,
            max_fetch_retries=max_fetch_retries
        )
    return _make


@pytest.fixture
def sample_tag_args(make_tag_args):
    """Create sample TagArgs for testing."""
    return [
        make_tag_args(feature="caption", stream="video"),
        make_tag_args(feature="asr", stream="audio"),
    ]