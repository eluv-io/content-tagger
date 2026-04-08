
import os
import shutil
import time
from unittest.mock import Mock
import uuid

from flask import Flask
import pytest

from app_config import AppConfig
from server import configure_routes, create_app_direct, create_app_queue_based
from src.api.auth import Authenticator
from src.api.tagging.request_format import StartJobsRequest
from src.common.content import Content
from src.fetch.model import DownloadResult, FetchSession, MediaMetadata, VideoScope
from src.service.model import StatusArgs, TagDetails, TagJobStatusResult
from src.status.get_info import UserInfoResolverConfig
from src.tag_containers.model import ModelConfig, RegistryConfig
from src.tagging.fabric_tagging.model import TaggerWorkerConfig, JobID, TagArgs, TagStartResult, TagStopResult
from src.tagging.fabric_tagging.queue.fs_jobstore import FsJobStore
from src.tagging.fabric_tagging.queue.model import JobStoreConfig
from src.tagging.fabric_tagging.tagger import TaggerWorker
from src.tagging.scheduling.model import SysConfig
from src.tagging.tag_runner import TagRunner, TagRunnerConfig
from src.tags.track_resolver import TrackArgs, TrackResolverConfig
from src.tags.tagstore.model import TagstoreConfig

@pytest.fixture()
def tagger_config(static_dir) -> TaggerWorkerConfig:
    media_path = os.path.join(static_dir, "media")
    os.makedirs(media_path, exist_ok=True)
    return TaggerWorkerConfig(media_dir=media_path)    

    
@pytest.fixture()
def container_registry_config(static_dir) -> RegistryConfig:
    return RegistryConfig(
            base_dir=os.path.join(static_dir, "stuff"),
            cache_dir=os.path.join(static_dir, "cache"),
            model_configs={
                "test_model": ModelConfig(
                    type="frame",
                    description="Test model",
                    resources={"gpu": 1},
                    image="localhost/test_model:latest"
                ),
                "test_model2": ModelConfig(
                    type="frame",
                    description="Test model 2",
                    resources={"gpu": 1},
                    image="localhost/test_model:latest"
                )
            }
        )

@pytest.fixture()
def app_config(static_dir, tagger_config, content_config, fetcher_config, container_registry_config) -> AppConfig:
    """Create test configuration."""
    return AppConfig(
        root_dir=static_dir,
        content=content_config,
        jobstore=JobStoreConfig(base_url=os.path.join(static_dir, "jobstore")),
        tagstore=TagstoreConfig(
            base_dir=os.path.join(static_dir, "tags")
        ),
        system=SysConfig(gpus=["gpu", "disabled", "gpu"], resources={"cpu_juice": 16}),
        fetcher=fetcher_config,
        container_registry=container_registry_config,
        tagger=tagger_config,
        track_resolver=TrackResolverConfig(mapping={"test_model": TrackArgs(name="test_model", label="TEST MODEL")}),
        tag_runner=TagRunnerConfig(poll_interval=0.1),
        user_info_resolver=UserInfoResolverConfig(
            fabric_url="https://main.net955305.contentfabric.io",
            user_info_url="https://ai.contentfabric.io/ml/token_info"
        )
    )


@pytest.fixture()
def app(static_dir, app_config):
    shutil.rmtree(static_dir, ignore_errors=True)
    if os.getenv("USE_QUEUE") == "true":
        app = create_app_queue_based(app_config)
    else:
        app = create_app_direct(app_config)
    app.config["TESTING"] = True
    yield app
    state = app.config["state"]
    if "loop" in state:
        state["loop"].stop()
        return
    tagger: TaggerWorker = state["worker"]
    if not tagger.shutdown_requested:
        tagger.cleanup()

@pytest.fixture
def authenticator(app_config):
    return Authenticator(app_config.content.config_url)

@pytest.fixture()
def client(app):
    return app.test_client()

class FakeLiveWorker(FetchSession):
    """Fake DownloadWorker that simulates live streaming by returning one source at a time"""
    
    def __init__(self, real_worker: FetchSession, last_res_has_media: bool=False):
        self.real_worker = real_worker
        self.call_count = 0
        self._all_sources = None
        self.last_res_has_media = last_res_has_media
    
    def metadata(self) -> MediaMetadata:
        return self.real_worker.metadata()
    
    @property 
    def path(self) -> str:
        return self.real_worker.path
    
    def download(self) -> DownloadResult:
        # Get all sources on first call
        if self._all_sources is None:
            real_result = self.real_worker.download()
            self._all_sources = real_result.sources
            self._failed = real_result.failed
        
        # Simulate live streaming delay
        time.sleep(2)

        self.call_count += 1

        idx = self.call_count - 1
        
        # Return one source at a time
        if idx < len(self._all_sources):
            # We have a source to return
            if idx == len(self._all_sources) - 1 and self.last_res_has_media:
                # Last source AND we want to include it with done=True
                return DownloadResult(
                    sources=[self._all_sources[idx]],
                    failed=self._failed,
                    done=True
                )
            else:
                # Not the last source, OR last source but we don't want done=True yet
                return DownloadResult(
                    sources=[self._all_sources[idx]],
                    failed=[],  # Don't report failures until the end
                    done=False
                )
        else:
            # No more sources - final empty call (only reached if last_res_has_media=False)
            return DownloadResult(
                sources=[],
                failed=self._failed,
                done=True
            )
        
class MockTaggerService:
    """In-memory mock implementation of TaggerService for use in tests."""

    def __init__(self):
        # job_id -> dict with job info
        self._jobs: dict[str, dict] = {}

    def tag(self, q: Content, args: TagArgs) -> TagStartResult:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "qid": q.qid,
            "status": "Tagging content",
            "model": args.feature,
            "stream": "",
            "created_at": time.time(),
            "params": args.run_config,
            "tenant": "test tenant",
            "user": "test user",
            "title": q.qid,
            "error": None,
        }
        return TagStartResult(
            job_id=JobID(qid=q.qid, feature=args.feature, stream=""),
            started=True,
            message="Mock job started",
        )

    def status(self, req: StatusArgs) -> list[TagJobStatusResult]:
        results = []
        for job in self._jobs.values():
            if req.qid and job["qid"] != req.qid:
                continue
            if req.tenant and job["tenant"] != req.tenant:
                continue
            if req.user and job["user"] != req.user:
                continue
            if req.title and job["title"] != req.title:
                continue
            details = TagDetails(
                tag_status=job["status"],
                time_running=time.time() - job["created_at"],
                progress=1.0 if job["status"] == "Completed" else 0.5,
                tagging_progress=job["status"],
                total_parts=1,
                downloaded_parts=1,
                tagged_parts=1 if job["status"] == "Completed" else 0,
                warnings=None,
            )
            results.append(TagJobStatusResult(
                qid=job["qid"],
                job_id=job["job_id"],
                status=job["status"],
                model=job["model"],
                stream=job["stream"],
                created_at=job["created_at"],
                params=job["params"],
                tenant=job["tenant"],
                user=job["user"],
                title=job["title"],
                error=job["error"],
                tagger_details=details,
            ))
        return results

    def stop(self, qid: str, feature: str | None) -> list[TagStopResult]:
        results = []
        for job in self._jobs.values():
            if job["qid"] != qid:
                continue
            if feature and job["model"] != feature:
                continue
            job["status"] = "Stopped"
            results.append(TagStopResult(job_id=job["job_id"], message="Mock job stopped"))
        return results

@pytest.fixture
def mock_tagger_service():
    return MockTaggerService()

@pytest.fixture
def mock_authenticator():
    class MockAuthenticator:
        def authenticate(self, q: Content) -> bool:
            return True
    return MockAuthenticator()

class MockArgsResolver:
    """Mock ArgsResolver with no external dependencies. Returns one TagArgs per job
    using a default VideoScope and the values directly from the request."""

    def resolve(self, args: StartJobsRequest, q: Content) -> list[TagArgs]:
        results = []
        for job in args.jobs:
            results.append(TagArgs(
                feature=job.model,
                run_config=job.model_params,
                scope=VideoScope(stream="video", start_time=0, end_time=10**16),
                replace=False,
                destination_qid="",
                max_fetch_retries=3,
            ))
        return results
    
@pytest.fixture
def mock_app(mock_tagger_service, mock_authenticator, fake_user_info_resolver):
    app = Flask(__name__)
    app.config["state"] = {
        "service": mock_tagger_service,
        "authenticator": mock_authenticator,
        "arg_resolver": MockArgsResolver(),
        "user_info_resolver": fake_user_info_resolver
    }
    configure_routes(app)
    return app