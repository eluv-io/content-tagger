import pytest
from unittest.mock import patch
import os
import shutil
import time
import json
from dotenv import load_dotenv
from src.common.logging import logger
from src.common.content import Content

from server import create_app
from app_config import AppConfig
import podman
from src.api.tagging.dto_mapping import _find_default_audio_stream
from src.common.content import ContentConfig, ContentFactory
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tags.conversion import TagConverterConfig
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.tags.tagstore.model import TagstoreConfig
from src.tagging.scheduling.model import SysConfig
from src.fetch.model import *
from src.tag_containers.model import ModelConfig, RegistryConfig
from src.tagging.fabric_tagging.model import FabricTaggerConfig

load_dotenv()

# Test configuration
test_objects = {
    "vod": "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm", 
    "assets": "iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2", 
}

def get_auth(qid: str) -> str:
    auth = None
    if qid == test_objects['vod']:
        auth = os.getenv(f"TEST_AUTH")
    elif qid == test_objects['assets']:
        auth = os.getenv(f"ASSETS_AUTH")
    assert auth is not None
    return auth

def get_content(auth: str, qhit: str):
    """Create Content object with write token from environment"""
    
    cfg = ContentConfig(
        config_url="https://host-154-14-185-98.contentfabric.io/config?self&qspace=main", 
        parts_url="https://host-154-14-185-98.contentfabric.io/config?self&qspace=main",
        live_media_url="https://host-76-74-34-204.contentfabric.io/config?self&qspace=main"
    )
    factory = ContentFactory(cfg=cfg)
    
    q = factory.create_content(qhit=qhit, auth=auth)

    return q

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

@pytest.fixture(scope="session")
def test_dir():
    test_dir = os.path.join(os.path.abspath(__file__), '../..', 'test-stuff')
    test_dir = os.path.abspath(test_dir)
    return test_dir

@pytest.fixture(scope="session")
def tagger_config(test_dir) -> FabricTaggerConfig:
    media_path = os.path.join(test_dir, "media")
    os.makedirs(media_path, exist_ok=True)
    return FabricTaggerConfig(media_dir=media_path)

@pytest.fixture(scope="session")
def test_config(test_dir, tagger_config):
    """Create test configuration."""
    return AppConfig(
        tag_converter=TagConverterConfig(
            interval=10,
            coalesce_tracks=[],
            single_tag_tracks=[],
            name_mapping={},
            max_sentence_words=100
        ),
        root_dir=test_dir,
        content=ContentConfig(
            config_url="https://main.net955305.contentfabric.io/config",
            parts_url="http://192.168.96.203/config?self&qspace=main",
            live_media_url="https://host-76-74-34-204.contentfabric.io/config?self&qspace=main"
        ),
        tagstore=TagstoreConfig(
            base_dir=os.path.join(test_dir, "tags")
        ),
        system=SysConfig(gpus=["gpu", "disabled", "gpu"], resources={"cpu_juice": 16}),
        fetcher=FetcherConfig(
            author="tagger",
            max_downloads=4
        ),
        container_registry=RegistryConfig(
            base_dir=os.path.join(test_dir, "stuff"),
            cache_dir=os.path.join(test_dir, "cache"),
            model_configs={
                "test_model": ModelConfig(
                    type="frame",
                    resources={"gpu": 1},
                    image="localhost/test_model:latest"
                )
            }
        ),
        tagger=tagger_config
    )


@pytest.fixture()
def app(test_dir, test_config):
    shutil.rmtree(test_dir, ignore_errors=True)
    app = create_app(test_config)
    app.config["TESTING"] = True
    yield app
    tagger: FabricTagger = app.config["state"]["tagger"]
    if not tagger.shutdown_requested:
        tagger.cleanup()

@pytest.fixture()
def client(app):
    return app.test_client()

def wait_for_jobs_completion(client, content_ids, timeout=30):
    """Wait for all jobs to complete."""
    start_time = time.time()
    if timeout is None:
        timeout = float('inf')
    while time.time() - start_time < timeout:
        all_finished = True
        
        for content_id in content_ids:
            auth = get_auth(content_id)
            response = client.get(f"/{content_id}/status?authorization={auth}")
            
            if response.status_code != 200:
                all_finished = False
                break
                
            result = response.get_json()
            print(json.dumps(result, indent=2))
            
            for stream in result:
                for feature in result[stream]:
                    status = result[stream][feature]['status']
                    if status not in ['Completed', 'Failed']:
                        all_finished = False
                        break
                if not all_finished:
                    break
        
        if all_finished:
            return True
            
        time.sleep(2)
    
    return False

@pytest.fixture(autouse=True)
def check_skip():
    # check if dummy_gpu & dummy_cpu are available images in podman
    with podman.PodmanClient() as client:
        images = sum([image.tags for image in client.images.list() if image.tags], [])
        if "localhost/test_model:latest" not in images:
            logger.warning("Test model image 'localhost/test_model:latest' not found.")
            pytest.skip("Required test images not found in local podman registry")

def test_video_model(client):
    """Test the complete tagging workflow."""
    # Get auth tokens
    qid = test_objects['vod']
    auth = get_auth(qid=qid)
    
    # Test initial status - should return 404 for no jobs
    response = client.get(f"/{qid}/status?authorization={auth}")
    assert response.status_code == 404
    
    # Start video tagging with GPU feature
    response = client.post(
        f"/{qid}/tag?authorization={auth}", 
        json={
            "features": {
                "test_model": {
                    "model": {"tags": ["hello1", "hello2"]}
                }, 
                #"shot": {}
            }, 
            "replace": True
        }
    )
    assert response.status_code == 200
    completed = wait_for_jobs_completion(client, [qid], timeout=30)
    assert completed
    tagstore: FilesystemTagStore = client.application.config["state"]["tagger"].tagstore
    jobid = tagstore.find_batches(q=get_content(auth, qid), stream='video')[0]
    tags = tagstore.find_tags(batch_id=jobid, q=get_content(auth, qid))
    tags = sorted(tags, key=lambda x: x.start_time)
    # TODO: this randomly gave 124 one time and I can't reproduce it
    assert len(tags) == 122
    next_tag = 'hello1'
    for tag in tags:
        assert tag.text == next_tag
        next_tag = 'hello2' if next_tag == 'hello1' else 'hello1'
        assert 'frame_tags' in tag.additional_info

    assert completed, "Timeout waiting for jobs to complete"

@pytest.mark.parametrize("last_res_has_media", [True, False])
@patch('src.api.tagging.dto_mapping._is_live')
def test_live_video_model(is_live, app, last_res_has_media):
    """Test the live tagging workflow with FakeLiveFetcher."""
    is_live.return_value = True
    # Get auth tokens
    qid = test_objects['vod']
    auth = get_auth(qid=qid)
    
    # Replace the real fetcher with FakeLiveFetcher
    tagger: FabricTagger = app.config["state"]["tagger"]
    tagstore = tagger.tagstore
    
    # Create FakeLiveFetcher with the same config
    original_get_worker = tagger.fetcher.get_session
    
    # Store reference to the FakeLiveWorker so we can access its call_count
    fake_worker_ref: list[FakeLiveWorker | None] = [None]
    
    # NOTE: ugliest thing i've ever seen
    def fake_get_worker(q: Content, req: DownloadRequest, exit=None) -> FetchSession:
        if fake_worker_ref[0] is not None:
            return fake_worker_ref[0]
        # we need the non-live worker in this test
        old_scope = req.scope
        req.scope = VideoScope(stream="video", start_time=0, end_time=float('inf'))
        real_worker = original_get_worker(q, req, exit)
        req.scope = old_scope
        fake_worker = FakeLiveWorker(real_worker, last_res_has_media)
        fake_worker_ref[0] = fake_worker  # Store reference
        return fake_worker
    
    # Replace with our version
    tagger.fetcher.get_session = fake_get_worker
    
    # Create test client
    client = app.test_client()
    
    # Test initial status - should return 404 for no jobs
    response = client.get(f"/{qid}/status?authorization={auth}")
    assert response.status_code == 404
    
    # Start video tagging with GPU feature
    response = client.post(
        f"/{qid}/tag?authorization={auth}", 
        json={
            "features": {
                "test_model": {
                    "model": {"tags": ["hello1", "hello2"]}
                }
            }, 
            "replace": True
        }
    )
    assert response.status_code == 200
    
    # Wait for job completion
    completed = wait_for_jobs_completion(client, [qid], timeout=60)
    assert completed, "Timeout waiting for live job to complete"
    
    # Calculate expected call count based on last_res_has_media
    expected_calls = 5 if last_res_has_media else 6
    
    # Verify the fetcher was called the expected number of times
    assert fake_worker_ref[0] is not None, "FakeLiveWorker was not created"
    assert fake_worker_ref[0].call_count == expected_calls, f"Expected {expected_calls} fetch calls, got {fake_worker_ref[0].call_count}"
    
    # Get tags and verify results
    jobid = tagstore.find_batches(q=get_content(auth, qid), stream='video')[0]
    tags = tagstore.find_tags(batch_id=jobid, q=get_content(auth, qid))
    tags = sorted(tags, key=lambda x: x.start_time)
    
    # Should have same number of tags as regular video model test
    assert len(tags) == 122, f"Expected 122 tags, got {len(tags)}"
    
    # Verify tags alternate between hello1 and hello2
    next_tag = 'hello1'
    for tag in tags:
        assert tag.text == next_tag, f"Expected {next_tag}, got {tag.text}"
        next_tag = 'hello2' if next_tag == 'hello1' else 'hello1'
        assert 'frame_tags' in tag.additional_info

    logger.info(f"Live test completed successfully with {fake_worker_ref[0].call_count} fetch calls (last_res_has_media={last_res_has_media})")

def test_real_live_stream(app, live_q):
    """Test real live stream tagging with LiveWorker."""
    qid = live_q.qid
    auth = live_q._client.token
    
    tagger: FabricTagger = app.config["state"]["tagger"]
    tagstore = tagger.tagstore
    
    # Create test client
    client = app.test_client()
    
    # Start live tagging with test_model
    # Use small chunk_size and max_duration for faster testing
    response = client.post(
        f"/{qid}/tag?authorization={auth}", 
        json={
            "features": {
                "test_model": {
                    "model": {"tags": ["hello1", "hello2"]}
                }
            },
            "max_duration": 20,
            "segment_length": 5,
            "replace": True
        }
    )
    assert response.status_code == 200
    
    # Wait for some segments to be processed (but not completion since it's live)
    # Check status periodically
    start_time = time.time()
    timeout = 25
    segments_found = False
    
    while time.time() - start_time < timeout:
        response = client.get(f"/{qid}/status?authorization={auth}")
        if response.status_code == 200:
            result = response.get_json()
            print(json.dumps(result, indent=2))
            
            # Check if we have the test_model job
            if 'video' in result and 'test_model' in result['video']:
                status = result['video']['test_model']['status']
                progress = result['video']['test_model']['tagging_progress']
                
                # Once we see progress or completion, we know segments were processed
                if progress == "100%" or status == "Completed":
                    segments_found = True
                    break
        
        time.sleep(3)
    
    #assert segments_found, "No segments were processed within timeout"
    
    # Verify final status is Stopped or Completed
    response = client.get(f"/{qid}/status?authorization={auth}")
    assert response.status_code == 200
    result = response.get_json()
    
    final_status = result['video']['test_model']['status']
    assert final_status in ['Stopped', 'Completed'], f"Expected Stopped or Completed, got {final_status}"
    
    # verify we have some tags
    jobid = tagstore.find_batches(q=live_q, stream='video', qhit=live_q.qid)[0]
    tags = tagstore.find_tags(batch_id=jobid, q=live_q)
    tags = sorted(tags, key=lambda x: x.start_time)
    
    # Should have at least some tags from the segments
    assert len(tags) > 0, "Expected 6 from live stream"
    
    logger.info(f"Live stream test completed successfully with {len(tags)} tags")

def test_asset_tag(client):
    """Test asset tagging."""
    qid = test_objects['assets']
    auth = get_auth(qid=qid)
    
    # Start asset tagging with CPU feature
    response = client.post(
        f"/{qid}/image_tag?authorization={auth}", 
        json={
            "features": {
                "test_model": {
                    "model": {"tags": ["hello world"]}
                }
            }
        }
    )
    assert response.status_code == 200
    completed = wait_for_jobs_completion(client, [qid], timeout=25)
    assert completed
    status = client.get(f"/{qid}/status?authorization={auth}")
    print(status.get_json())
    tagstore: FilesystemTagStore = client.application.config["state"]["tagger"].tagstore
    jobid = tagstore.find_batches(qhit=test_objects['assets'], stream='assets')[0]
    tags = tagstore.find_tags(batch_id=jobid)
    tags = sorted(tags, key=lambda x: x.start_time)
    assert len(tags) > 0

def test_stop_workflow(client):
    """Test stopping jobs."""
    video_auth = get_auth(qid=test_objects['vod'])
    
    # Start a job
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={video_auth}", 
        json={
            "features": {
                "test_model": {
                    "model": {"tags": ["test_stop"]}
                }
            }, 
            "replace": True
        }
    )
    assert response.status_code == 200
    
    # Stop the job quickly (before it completes)
    response = client.post(f"/{test_objects['vod']}/stop/test_model?authorization={video_auth}")
    #assert response.status_code == 200
    
    # Check status - job should be stopped
    response = client.get(f"/{test_objects['vod']}/status?authorization={video_auth}")
    assert response.status_code == 200
    result = response.get_json()
    print('asdffsdf')
    print(json.dumps(result, indent=2))
    
    # The job should exist and be in a stopped state
    assert 'video' in result
    assert 'test_model' in result['video']
    status = result['video']['test_model']['status']
    assert status == 'Stopped', f"Expected job to be stopped, got {status}"

def test_double_run(client):
    """Run same job twice, expect second to be rejected."""
    video_auth = get_auth(qid=test_objects['vod'])
    
    # Start initial job
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={video_auth}", 
        json={
            "features": {
                "test_model": {
                    "model": {"tags": ["original_tags"]}
                }
            }
        }
    )
    assert response.status_code == 200
    
    # Try to start another job with replace=False (should be rejected)
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={video_auth}", 
        json={
            "features": {
                "test_model": {
                    "model": {"tags": ["should_not_replace"]}
                }
            },
        }
    )
    data = response.get_json()
    assert response.status_code == 200
    assert "already running" in data["test_model"]

    # stop the job
    start = time.time()
    response = client.post(f"/{test_objects['vod']}/stop/test_model?authorization={video_auth}")
    duration = time.time() - start
    print(duration)
    assert duration < 2, f"Stop request took too long: {duration}s which is over 2s limit"
    assert response.status_code == 200

def test_find_default_audio_stream(
):
    q = get_content(get_auth(test_objects['vod']), test_objects['vod'])
    result = _find_default_audio_stream(q)

    assert result == "stereo"

def test_is_live(live_q):
    """Test the _is_live function."""
    from src.api.tagging.dto_mapping import _is_live
    assert _is_live(live_q) == True

def test_stop_live_job(app, live_q):
    """Test that live jobs can be stopped cleanly mid-stream."""
    qid = live_q.qid
    auth = live_q._client.token
    
    tagger: FabricTagger = app.config["state"]["tagger"]
    tagstore = tagger.tagstore
    client = app.test_client()
    
    # Start live tagging with long duration
    response = client.post(
        f"/{qid}/tag?authorization={auth}", 
        json={
            "features": {"test_model": {"model": {"tags": ["hello1", "hello2"]}}},
            "max_duration": 60,  # Long duration so it won't complete
            "segment_length": 4,
            "replace": True
        }
    )
    assert response.status_code == 200
    
    # Wait for job to start processing and get some tags
    start_time = time.time()
    some_tags_found = False
    
    while time.time() - start_time < 20:
        response = client.get(f"/{qid}/status?authorization={auth}")
        if response.status_code == 200:
            result = response.get_json()
            if 'video' in result and 'test_model' in result['video']:
                progress = result['video']['test_model']['tagging_progress']
                
                # Wait until we have some progress but not complete
                # Parse progress like "50%" -> 50
                if progress != "0%" and progress != "100%":
                    # Check we actually have some tags
                    try:
                        jobid = tagstore.find_batches(q=live_q, stream='video', qhit=live_q.qid)[0]
                        tags = tagstore.find_tags(batch_id=jobid, q=live_q)
                        if len(tags) > 0:
                            some_tags_found = True
                            logger.info(f"Found {len(tags)} tags, stopping job now")
                            break
                    except IndexError:
                        pass  # No tags yet, keep waiting
        time.sleep(2)
    
    assert some_tags_found, "No tags were processed before stop attempt"
    
    # Stop the job
    stop_start = time.time()
    response = client.post(f"/{qid}/stop/test_model?authorization={auth}")
    stop_duration = time.time() - stop_start
    
    assert response.status_code == 200
    assert stop_duration < 3, f"Stop took {stop_duration}s (expected < 3s)"
    
    # Verify it stopped
    time.sleep(2)
    response = client.get(f"/{qid}/status?authorization={auth}")
    result = response.get_json()
    
    final_status = result['video']['test_model']['status']
    assert final_status == 'Stopped', f"Expected Stopped, got {final_status}"
    
    # Verify we have partial tags (not all of them)
    jobid = tagstore.find_batches(q=live_q, stream='video', qhit=live_q.qid)[0]
    final_tags = tagstore.find_tags(batch_id=jobid, q=live_q)
    
    assert len(final_tags) > 0, "Should have some tags"
    assert len(final_tags) < 20, f"Should have partial tags, got {len(final_tags)} (too many, job may have completed)"
    
    logger.info(f"Live stop test completed with {len(final_tags)} partial tags")