
import os
import shutil
import time

import pytest

from app_config import AppConfig
from server import create_app
from src.fetch.model import DownloadResult, FetchSession, MediaMetadata
from src.tag_containers.model import ModelConfig, RegistryConfig
from src.tagging.fabric_tagging.model import FabricTaggerConfig
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tagging.scheduling.model import SysConfig
from src.tagging.uploading.config import ModelUploadArgs, TrackArgs, UploaderConfig
from src.tags.conversion import TagConverterConfig
from src.tags.tagstore.model import TagstoreConfig

@pytest.fixture()
def tagger_config(static_dir) -> FabricTaggerConfig:
    media_path = os.path.join(static_dir, "media")
    os.makedirs(media_path, exist_ok=True)
    return FabricTaggerConfig(media_dir=media_path, uploader=UploaderConfig(model_params={"test_model": ModelUploadArgs(default=TrackArgs(name="test_model", label="TEST MODEL"))}))    

    
@pytest.fixture()
def container_registry_config(static_dir) -> RegistryConfig:
    return RegistryConfig(
            base_dir=os.path.join(static_dir, "stuff"),
            cache_dir=os.path.join(static_dir, "cache"),
            model_configs={
                "test_model": ModelConfig(
                    type="frame",
                    resources={"gpu": 1},
                    image="localhost/test_model:latest"
                )
            }
        )

@pytest.fixture()
def app_config(static_dir, tagger_config, content_config, fetcher_config, container_registry_config) -> AppConfig:
    """Create test configuration."""
    return AppConfig(
        tag_converter=TagConverterConfig(
            interval=10,
            coalesce_tracks=[],
            single_tag_tracks=[],
            name_mapping={},
            max_sentence_words=100
        ),
        root_dir=static_dir,
        content=content_config,
        tagstore=TagstoreConfig(
            base_dir=os.path.join(static_dir, "tags")
        ),
        system=SysConfig(gpus=["gpu", "disabled", "gpu"], resources={"cpu_juice": 16}),
        fetcher=fetcher_config,
        container_registry=container_registry_config,
        tagger=tagger_config
    )


@pytest.fixture()
def app(static_dir, app_config):
    shutil.rmtree(static_dir, ignore_errors=True)
    app = create_app(app_config)
    app.config["TESTING"] = True
    yield app
    tagger: FabricTagger = app.config["state"]["tagger"]
    if not tagger.shutdown_requested:
        tagger.cleanup()

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