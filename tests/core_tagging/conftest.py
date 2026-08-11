import os
import threading
import time
from unittest.mock import Mock
import pytest

from src.common.content import Content
from src.fetch.model import DownloadRequest, DownloadResult, FetchSession, MediaMetadata, Source, VideoMetadata, VideoScope
from src.fetch.model import VideoScope
from src.service.job_poster import JobPoster
from src.service.model import StatusArgs
from src.status.get_info import UserInfo
from src.tag_containers.model import *
from src.tagging.fabric_tagging.model import TaggerWorkerConfig, TagArgs
from src.tagging.fabric_tagging.source_resolver import SourceResolver
from src.tagging.fabric_tagging.tagger import TaggerWorker
from src.tagging.fabric_tagging.queue.fs_jobstore import FsJobStore
from src.tagging.tag_runner import TagRunner, TagRunnerConfig
from src.service.impl.queue_based import QueueService
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.tagging.scheduling.model import SysConfig
from src.tags.track_resolver import TrackArgs, TrackResolver, LabelResolverConfig


@pytest.fixture
def media_dir(temp_dir: str) -> str:
    """Create a media directory for testing"""
    media_path = os.path.join(temp_dir, "media")
    os.makedirs(media_path, exist_ok=True)
    return media_path

@pytest.fixture
def model_configs():
    return {
        "caption": Mock(type="video", track_outputs=["object_detection"], track_dependencies=[]),
        "asr": Mock(type="audio", track_outputs=["speech_to_text"], track_dependencies=[]),
    }

@pytest.fixture
def track_resolver(model_configs):
    """Create a simple track resolver for testing"""
    return TrackResolver(
        label_configs=LabelResolverConfig(mapping={
            "object_detection": "Object Detection",
            "speech_to_text": "Speech to Text",
            "pretty": "Pretty Speech"
        }),
        model_configs=model_configs
    )
    
@pytest.fixture
def tagger_config(media_dir) -> TaggerWorkerConfig:
    return TaggerWorkerConfig(
        media_dir=media_dir,
    )

class FakeTagContainer:
    """Fake TagContainer that simulates work and asynchronous behavior."""
    
    def __init__(
        self, 
        media_dir: str, 
        feature, 
        work_duration: float = 0.5, 
        report_progress: bool = True,
        output_vectors: bool = False
    ):
        """
        Initialize the FakeTagContainer.
    
        Args:
            media: List of file paths or directory to process.
            feature (str): The feature being tagged.
            work_duration (float): Time in seconds to simulate work.
        """
        self.report_progress = report_progress
        self.media_dir = media_dir
        self.feature = feature
        self.work_duration = work_duration
        self.is_started = False
        self.is_stopped = False
        self.container = Mock()
        self.container.attrs = {"State": {"ExitCode": 0}}
        self.tag_call_count = 0
        self.media_files = []
        self.worker_thread = None
        self._tag_cursor = 0
        self._output_vectors = output_vectors

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

    def info(self) -> ContainerInfo:
        return ContainerInfo(image_name=f"fake/{self.feature}", annotations={"io.test.fake": "1"})
    
    def add_media(self, new_media: list[str]) -> None:
        self.media_files += new_media

    def new_tags(self) -> list[ModelTag]:
        all_tags = self._tags()
        start = self._tag_cursor
        self._tag_cursor = len(all_tags)
        return all_tags[start:]

    def _tags(self) -> list[ModelTag]:
        """
        Return fake tags for the media files.

        Returns:
            list[ModelTag]: List of fake tags for each file.
        """
        self.tag_call_count += 1

        tags = []
        if not self.is_stopped:
            finished_files = self.media_files[:-1]
        else:
            finished_files = self.media_files

        for i, filepath in enumerate(finished_files):
            # Create fake tags based on the feature
            if not self._output_vectors:
                data1 = f"{self.feature}_tag_{i}"
                data2 = f"{self.feature}_tag_{i}_2"
            else:
                data1 = [i, i + 0.1, i + 0.2]
                data2 = [i + 0.3, i + 0.4, i + 0.5]

            fake_tags = [
                ModelTag(
                    start_time=0,
                    end_time=5000,  # 5 seconds in ms
                    data=data1,
                    source_media=filepath,
                    model_track=""
                ),
                ModelTag(
                    start_time=5000,
                    end_time=10000,  # 5-10 seconds in ms
                    data=data2,
                    source_media=filepath,
                    model_track=""
                )
            ]

            tags.extend(fake_tags)

        return tags
    
    def progress(self) -> list[Progress]:
        finished_files = self.media_files[:-1] if not self.is_stopped else self.media_files
        return [Progress(source_media=filepath) for filepath in finished_files]
    
    def progress_ratio(self) -> float | None:
        if self.report_progress:
            if not self.media_files:
                return 1.0
            finished_files = self.media_files[:-1] if not self.is_stopped else self.media_files
            return (len(finished_files) / len(self.media_files))
        return None
    
    def errors(self) -> list[Error]:
        return []
    
    def name(self) -> str:
        return f"FakeContainer-{self.feature}"
    
    def required_resources(self):
        return {}
    
    def is_content_aligned(self) -> bool:
        return False

class PartialResultContainer(FakeTagContainer):
    """
    Don't return tag for last source
    """

    def tags(self) -> list[ModelTag]:
        tags = super().new_tags()
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
        container_key = f"{req.model_id}_{req.media_dir}"
        if container_key not in self.containers:
            self.containers[container_key] = FakeTagContainer(req.media_dir, req.model_id)
        return self.containers[container_key]
    
    def get_model_config(self, feature: str) -> ModelConfig:
        if feature == "caption":
            return Mock(resources={"gpu": 1, "cpu_juice": 5}, tags=Mock(hidden=False))
        elif feature == "asr":
            return Mock(resources={"gpu": 1, "cpu_juice": 5}, tags=Mock(hidden=True))
        return Mock(resources={"gpu": 1, "cpu_juice": 5})
    
    def services(self) -> list[str]:
        return ["caption", "asr"]
    
    @property
    def model_configs(self):
        model_configs = {
            "caption": Mock(type="video"),
            "asr": Mock(type="audio"),
        }
        return model_configs

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
        self._output_meta = MediaMetadata(
            sources=self._metadata.parts,
            fps=self._metadata.fps
        )

    def metadata(self) -> MediaMetadata:
        return self._output_meta

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
                name=f"hash{i+1}",
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
    
class NoopWorker(FetchSession):
    def metadata(self) -> MediaMetadata:
        return MediaMetadata(sources=[], fps=None)
    
    def download(self) -> DownloadResult:
        return DownloadResult(sources=[], failed=[], done=True)
    
    @property
    def path(self) -> str:
        return ""

@pytest.fixture
def fake_fetcher(media_dir):
    """Create a fake fetcher that returns FakeWorker"""
    class FakeFetcher:
        def __init__(self):
            self.config = Mock(author="tagger", max_downloads=2)

        def get_session(self, q: Content, req: DownloadRequest, exit_event=None):
            """Return a FakeWorker"""
            if req.ignore_sources:
                # assume we are testing replace functionality and just return noop
                return NoopWorker()
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
def source_resolver(tag_store, vectorstores, track_resolver):
    """Create a SourceResolver using the provided stores and track resolver."""
    return SourceResolver(tagstore=tag_store, vectorstores=vectorstores, track_resolver=track_resolver)


@pytest.fixture
def fabric_tagger(system_tagger, fake_container_registry, tag_store, vectorstores, fake_fetcher, track_resolver, tagger_config, source_resolver):
    """Create a TaggerWorker instance for testing"""
    tagger = TaggerWorker(
        system_tagger=system_tagger,
        cregistry=fake_container_registry,
        tagstore=tag_store,
        vectorstores=vectorstores,
        fetcher=fake_fetcher,
        track_resolver=track_resolver,
        cfg=tagger_config,
        source_resolver=source_resolver
    )
    yield tagger
    if not tagger.shutdown_requested():
        tagger.cleanup()

@pytest.fixture
def sample_tag_args(make_tag_args):
    """Create sample TagArgs for testing."""
    return [
        make_tag_args(feature="caption", stream="video"),
        make_tag_args(feature="asr", stream="audio"),
    ]

@pytest.fixture
def queue_jobstore(tmp_path, fake_user_info_resolver) -> FsJobStore:
    return FsJobStore(store_dir=str(tmp_path / "jobstore"), user_info_resolver=fake_user_info_resolver)

@pytest.fixture
def simple_job_poster(queue_jobstore, track_resolver, model_configs, fake_qapifactory) -> JobPoster:
    """Create a simple JobPoster for testing."""
    return JobPoster(
        job_store=queue_jobstore,
        track_resolver=track_resolver,
        model_configs=model_configs,
        qfactory=fake_qapifactory
    )

@pytest.fixture
def fake_qapifactory():
    # for the queue client, all we need is to get the display title and add this to the job info
    return Mock(
        create=Mock(
            return_value=Mock(
                content_object_metadata=Mock(
                    return_value="test-title"
                )
            )
        )
    )

@pytest.fixture
def job_poster(queue_jobstore, track_resolver, fake_qapifactory, model_configs) -> JobPoster:
    """Create a JobPoster for testing, using the queue_jobstore and other dependencies."""
    return JobPoster(
        job_store=queue_jobstore,
        track_resolver=track_resolver,
        model_configs=model_configs,
        qfactory=fake_qapifactory
    )


@pytest.fixture
def queue_client(job_poster) -> QueueService:
    return QueueService(job_poster=job_poster)


@pytest.fixture
def tag_runner(fabric_tagger, queue_jobstore, qfactory):
    """A TagRunner wired to the same FsJobStore, polling fast for tests."""
    runner = TagRunner(
        tagger=fabric_tagger,
        jobstore=queue_jobstore,
        cfg=TagRunnerConfig(poll_interval=0.1, max_jobs=2),
    )
    runner.start()
    yield runner
    if not runner._shutdown.is_set():
        runner.stop()

@pytest.fixture
def make_status_args():
    def _make(
        qid: str | None=None, 
        user: str | None=None,
        tenant: str | None=None,
        title: str | None=None
    ):
        return StatusArgs(
            qid=qid,
            user=user,
            tenant=tenant,
            title=title
        )
    
    return _make