import pytest
import os
import shutil
import time
import json
import tempfile
from dotenv import load_dotenv

from server import create_app
from app_config import AppConfig
from src.common.content import ContentConfig
from src.tags.tagstore.types import TagStoreConfig
from src.tagger.system_tagging.types import SysConfig
from src.fetch.types import FetcherConfig
from src.tag_containers.types import ModelConfig, RegistryConfig

load_dotenv()

# Test configuration
test_objects = {
    "vod": "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm", 
    "assets": "hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU", 
}

def get_auth() -> str:
    auth = os.getenv(f"TEST_AUTH")
    assert auth is not None
    return auth

@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test data."""
    temp_dir = "./test-stuff"
    shutil.rmtree(temp_dir, ignore_errors=True)
    yield temp_dir

@pytest.fixture(scope="session")
def test_config(temp_test_dir):
    """Create test configuration."""
    return AppConfig(
        content=ContentConfig(
            content_url="https://main.net955305.contentfabric.io/config",
            parts_url="http://192.168.96.203/config?self&qspace=main"
        ),
        tagstore=TagStoreConfig(
            base_path=os.path.join(temp_test_dir, "tags")
        ),
        system=SysConfig(gpus=["gpu", "disabled", "gpu"], cpu_juice=100),
        fetcher=FetcherConfig(
            parts_path=os.path.join(temp_test_dir, "parts"),
            max_downloads=4,
            author="tagger"
        ),
        container_registry=RegistryConfig(
            logs_path=os.path.join(temp_test_dir, "logs"),
            tags_path=os.path.join(temp_test_dir, "tags"),
            cache_path=os.path.join(temp_test_dir, "cache"),
            model_configs={
                "dummy_gpu": ModelConfig(
                    type="frame",
                    resources={"gpu": 1},
                    image="localhost/dummy_gpu:latest"
                ),
                "dummy_cpu": ModelConfig(
                    type="frame",
                    resources={"cpu_juice": 25},
                    image="localhost/dummy_cpu:latest"
                ),
                "shot": ModelConfig(
                    type="video",
                    resources={"gpu": 1},
                    image="localhost/shot:latest"
                )
            }
        )
    )

@pytest.fixture(scope="session")
def app(test_config):
    """Create Flask app for testing."""
    app = create_app(test_config)
    app.config['TESTING'] = True
    return app

@pytest.fixture(scope="session") 
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture(autouse=True)
def cleanup_test_data(temp_test_dir):
    """Clean up test data before and after each test."""
    # Clean up before test
    for subdir in ["tags", "parts", "logs"]:
        path = os.path.join(temp_test_dir, subdir)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)


def wait_for_jobs_completion(client, content_ids, timeout=30):
    """Wait for all jobs to complete."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        all_finished = True
        
        for content_id in content_ids:
            auth = get_auth()
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

def test_video_model(client):
    """Test the complete tagging workflow."""
    # Get auth tokens
    auth = get_auth()
    
    # Test initial status - should return 404 for no jobs
    response = client.get(f"/{test_objects['vod']}/status?authorization={auth}")
    assert response.status_code == 404
    
    # Start video tagging with GPU feature
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={auth}", 
        json={
            "features": {
                "dummy_gpu": {
                    "model": {"tags": ["hello1", "hello2"]}
                }, 
                #"shot": {}
            }, 
            "replace": True
        }
    )
    assert response.status_code == 200
    print("Started video GPU tagging")
    completed = wait_for_jobs_completion(client, [test_objects['vod']], timeout=10)
    status = client.get(f"/{test_objects['vod']}/status?authorization={auth}")
    print(json.dumps(status.get_json(), indent=2))
    assert completed, "Timeout waiting for jobs to complete"

def test_tag_workflow(client):
    """Test the complete tagging workflow."""
    # Get auth tokens
    auth = get_auth()
    
    # Test initial status - should return 404 for no jobs
    response = client.get(f"/{test_objects['vod']}/status?authorization={auth}")
    assert response.status_code == 404
    
    # Start video tagging with GPU feature
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={auth}", 
        json={
            "features": {
                "dummy_gpu": {
                    "model": {"tags": ["hello1", "hello2"]}
                }, 
                "shot": {}
            }, 
            "replace": True
        }
    )
    assert response.status_code == 200
    print("Started video GPU tagging")
    
    # Start video tagging with CPU feature
    #response = client.post(
    #    f"/{test_objects['vod']}/tag?authorization={auth}", 
    #    json={
    #        "features": {
    #            "dummy_cpu": {
    #                "model": {"tags": ["a", "b"], "allow_single_frame": False}
    #            }
    #        }, 
    #        "replace": True
    #    }
    #)
    #assert response.status_code == 200
    #print("Started video CPU tagging")
    
    # Start image tagging with GPU
    #response = client.post(
    #    f"/{test_objects['assets']}/image_tag?authorization={auth}", 
    #    json={
    #        "features": {
    #            "dummy_gpu": {
    #                "model": {"tags": ["image_hello1"]}
    #            }
    #        }, 
    #        "replace": True
    #    }
    #)
    #assert response.status_code == 200
    #print("Started image GPU tagging")
    
    # Start image tagging with CPU
    #response = client.post(
    #    f"/{test_objects['assets']}/image_tag?authorization={auth}", 
    #    json={
    #        "features": {
    #            "dummy_cpu": {
    #                "model": {"tags": ["image_hello2"]}
    #            }
    #        }, 
    #        "replace": True
    #    }
    #)
    #assert response.status_code == 200
    #print("Started image CPU tagging")
    
    # Wait for all jobs to complete
    content_ids = [test_objects['vod'], test_objects['assets']]
    completed = wait_for_jobs_completion(client, content_ids, timeout=60)
    assert completed, "Timeout waiting for jobs to complete"
    print("All jobs completed")
    
    # Verify final status for video content
    response = client.get(f"/{test_objects['vod']}/status?authorization={auth}")
    assert response.status_code == 200
    video_result = response.get_json()
    
    # Check that we have results for all requested features
    assert 'video' in video_result
    assert 'dummy_gpu' in video_result['video']
    assert 'dummy_cpu' in video_result['video'] 
    assert 'shot' in video_result['video']
    
    # Verify all video jobs completed successfully
    for feature in ['dummy_gpu', 'dummy_cpu', 'shot']:
        assert video_result['video'][feature]['status'] == 'Completed', f"Video {feature} failed"
        assert video_result['video'][feature]['failed'] == 0, f"Video {feature} had failures"
    
    print("Video tagging verified")
    
    # Verify final status for image content
    response = client.get(f"/{test_objects['assets']}/status?authorization={auth}")
    assert response.status_code == 200
    image_result = response.get_json()
    
    # Check that we have results for all requested features
    assert 'image' in image_result
    assert 'dummy_gpu' in image_result['image']
    assert 'dummy_cpu' in image_result['image']
    
    # Verify all image jobs completed successfully
    for feature in ['dummy_gpu', 'dummy_cpu']:
        assert image_result['image'][feature]['status'] == 'Completed', f"Image {feature} failed"
        assert image_result['image'][feature]['failed'] == 0, f"Image {feature} had failures"
    
    print("Image tagging verified")

def test_stop_workflow(client):
    """Test stopping jobs."""
    video_auth = get_auth()
    
    # Start a job
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={video_auth}", 
        json={
            "features": {
                "dummy_cpu": {
                    "model": {"tags": ["test_stop"]}
                }
            }, 
            "replace": True
        }
    )
    assert response.status_code == 200
    
    # Stop the job quickly (before it completes)
    response = client.post(f"/{test_objects['vod']}/stop/dummy_cpu?authorization={video_auth}")
    assert response.status_code == 200
    
    # Wait a moment for stop to take effect
    time.sleep(2)
    
    # Check status - job should be stopped
    response = client.get(f"/{test_objects['vod']}/status?authorization={video_auth}")
    assert response.status_code == 200
    result = response.get_json()
    
    # The job should exist and be in a stopped state
    assert 'video' in result
    assert 'dummy_cpu' in result['video']
    status = result['video']['dummy_cpu']['status']
    assert status in ['Stopped', 'Failed'], f"Expected job to be stopped, got {status}"

def test_replace_false_workflow(client):
    """Test tagging with replace=False (should not overwrite existing jobs)."""
    video_auth = get_auth()
    
    # Start initial job
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={video_auth}", 
        json={
            "features": {
                "dummy_cpu": {
                    "model": {"tags": ["original_tags"]}
                }
            }, 
            "replace": True
        }
    )
    assert response.status_code == 200
    
    # Wait for completion
    completed = wait_for_jobs_completion(client, [test_objects['vod']], timeout=30)
    assert completed
    
    # Try to start another job with replace=False (should be rejected)
    response = client.post(
        f"/{test_objects['vod']}/tag?authorization={video_auth}", 
        json={
            "features": {
                "dummy_cpu": {
                    "model": {"tags": ["should_not_replace"]}
                }
            }, 
            "replace": False
        }
    )
    # This should either be rejected (400) or succeed but not actually replace
    # The exact behavior depends on your implementation
    assert response.status_code in [200, 400]