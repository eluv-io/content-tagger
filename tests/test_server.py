import pytest
import os
import shutil
import time
import json
from dotenv import load_dotenv
from loguru import logger

from server import create_app
from app_config import AppConfig
import podman
from src.common.content import ContentConfig
from src.tagger.fabric_tagging.tagger import FabricTagger
from src.tags.conversion import TagConverterConfig
from src.tags.tagstore.tagstore import FilesystemTagStore
from src.tags.tagstore.types import TagStoreConfig
from src.tagger.system_tagging.types import SysConfig
from src.fetch.types import FetcherConfig
from src.tag_containers.types import ModelConfig, RegistryConfig

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

@pytest.fixture(scope="session")
def test_dir():
    test_dir = os.path.join(os.path.abspath(__file__), '../..', 'test-stuff')
    test_dir = os.path.abspath(test_dir)
    return test_dir

@pytest.fixture(scope="session")
def test_config(test_dir):
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
            parts_url="http://192.168.96.203/config?self&qspace=main"
        ),
        tagstore=TagStoreConfig(
            base_dir=os.path.join(test_dir, "tags")
        ),
        system=SysConfig(gpus=["gpu", "disabled", "gpu"], resources={"cpu_juice": 16}),
        fetcher=FetcherConfig(
            parts_dir=os.path.join(test_dir, "parts"),
            max_downloads=4,
            author="tagger"
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
        )
    )

@pytest.fixture() 
def client(test_dir, test_config):
    """Create Flask app for testing."""
    shutil.rmtree(test_dir, ignore_errors=True)
    app = create_app(test_config)
    app.config['TESTING'] = True
    yield app.test_client()
    tagger: FabricTagger = app.config["state"]["tagger"]
    if tagger.shutdown_requested is False:
        tagger.cleanup()

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
    auth = get_auth(qid=test_objects['vod'])
    
    # Test initial status - should return 404 for no jobs
    response = client.get(f"/{test_objects['vod']}/status?authorization={auth}")
    assert response.status_code == 404
    
    # Start video tagging with GPU feature
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={auth}", 
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
    completed = wait_for_jobs_completion(client, [test_objects['vod']], timeout=None)
    assert completed
    tagstore: FilesystemTagStore = client.application.config["state"]["tagger"].tagstore
    jobid = tagstore.find_jobs(qhit=test_objects['vod'], stream='video')[0]
    tags = tagstore.get_tags(jobid)
    tags = sorted(tags, key=lambda x: x.start_time)
    assert len(tags) == 122
    next_tag = 'hello1'
    for tag in tags:
        assert tag.text == next_tag
        next_tag = 'hello2' if next_tag == 'hello1' else 'hello1'
        assert 'frame_tags' in tag.additional_info

    assert completed, "Timeout waiting for jobs to complete"

def test_asset_tag(client):
    """Test asset tagging."""
    auth = get_auth(qid=test_objects['assets'])
    
    # Start asset tagging with CPU feature
    response = client.post(
        f"/{test_objects['assets']}/image_tag?authorization={auth}", 
        json={
            "features": {
                "test_model": {
                    "model": {"tags": ["hello world"]}
                }
            }
        }
    )
    assert response.status_code == 200
    completed = wait_for_jobs_completion(client, [test_objects['assets']], timeout=25)
    assert completed
    status = client.get(f"/{test_objects['assets']}/status?authorization={auth}")
    print(status.get_json())
    tagstore: FilesystemTagStore = client.application.config["state"]["tagger"].tagstore
    jobid = tagstore.find_jobs(qhit=test_objects['assets'], stream='assets')[0]
    tags = tagstore.get_tags(jobid)
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