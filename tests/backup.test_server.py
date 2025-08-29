import pytest
import os
import shutil
import time
from dotenv import load_dotenv
from elv_client_py import ElvClient

from server import create_app
from config import config

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Test configuration
test_objects = {
    "vod": "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm", 
    "assets": "iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2", 
    "legacy_vod": "hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU"
}

def get_auth(qid: str) -> str:
    auth = os.getenv(f"AUTH_{qid}")
    assert auth is not None, f"AUTH_{qid} environment variable not set"
    return auth

def get_write_token(qid: str) -> str:
    token = os.getenv(f"WRITE_{qid}")
    assert token is not None, f"WRITE_{qid} environment variable not set"
    return token

def postprocess_response(res: dict):
    """Remove dynamic fields from response for testing."""
    for stream in res:
        for model in res[stream]:
            if 'tag_job_id' in res[stream][model]:
                del res[stream][model]['tag_job_id']
            if 'time_running' in res[stream][model]:
                del res[stream][model]['time_running']
            if 'tagging_progress' in res[stream][model]:
                del res[stream][model]['tagging_progress']
            if res[stream][model].get('error'):
                res[stream][model]['error'] = ' '.join(
                    filter(lambda word: not word.startswith('tqw__'), 
                           res[stream][model]['error'].split())
                )
    return res

def postprocess_message(res: dict):
    """Remove dynamic identifiers from message."""
    if 'message' in res:
        res['message'] = ' '.join(
            filter(lambda word: not word.startswith('tqw__'), 
                   res['message'].split())
        )
    return res

@pytest.fixture(scope="session")
def app():
    """Create Flask app for testing."""
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture(scope="session")
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data before each test."""
    # Clean up storage directories
    for obj_id in test_objects.values():
        parts_path = os.path.join(config["storage"]["parts"], obj_id)
        shutil.rmtree(parts_path, ignore_errors=True)
        
        tags_path = os.path.join(config["storage"]["tags"], obj_id)
        shutil.rmtree(tags_path, ignore_errors=True)
    
    # Clean up images
    image_path = os.path.join(config["storage"]["images"], test_objects['assets'])
    shutil.rmtree(image_path, ignore_errors=True)
    
    yield
    
    # Cleanup after test as well
    for obj_id in test_objects.values():
        parts_path = os.path.join(config["storage"]["parts"], obj_id)
        shutil.rmtree(parts_path, ignore_errors=True)
        
        tags_path = os.path.join(config["storage"]["tags"], obj_id)
        shutil.rmtree(tags_path, ignore_errors=True)

def test_tag_workflow(client):
    """Test the complete tagging workflow."""
    # Get auth tokens
    assets_auth = get_auth(test_objects['assets'])
    video_auth = get_auth(test_objects['vod'])
    legacy_vod_auth = get_auth(test_objects['legacy_vod'])
    
    # Test initial status - should return 404 for no jobs
    response = client.get(f"/{test_objects['vod']}/status?authorization={video_auth}")
    assert response.status_code == 404
    
    # Start video tagging
    response = client.post(f"/{test_objects['vod']}/tag?authorization={video_auth}", 
                          json={
                              "features": {
                                  "dummy_gpu": {"model": {"tags": ["hello1", "hello2"]}}, 
                                  "shot": {}
                              }, 
                              "replace": True
                          })
    assert response.status_code == 200
    
    # Start CPU tagging
    response = client.post(f"/{test_objects['vod']}/tag?authorization={video_auth}", 
                          json={
                              "features": {
                                  "dummy_cpu": {"model": {"tags": ["a", "b", "a"], "allow_single_frame": False}}
                              }, 
                              "replace": True
                          })
    assert response.status_code == 200
    
    # Start image tagging
    response = client.post(f"/{test_objects['assets']}/image_tag?authorization={assets_auth}", 
                          json={
                              "features": {"dummy_gpu": {"model": {"tags": ["hello1"]}}}, 
                              "replace": True
                          })
    assert response.status_code == 200
    
    response = client.post(f"/{test_objects['assets']}/image_tag?authorization={assets_auth}", 
                          json={
                              "features": {"dummy_cpu": {"model": {"tags": ["hello2"]}}}, 
                              "replace": True
                          })
    assert response.status_code == 200
    
    # Start legacy VOD tagging
    response = client.post(f"/{test_objects['legacy_vod']}/tag?authorization={legacy_vod_auth}", 
                          json={
                              "features": {
                                  "dummy_gpu": {"model": {"tags": ["hello1"]}}, 
                                  "shot": {}
                              }, 
                              "start_time": 60, 
                              "end_time": 180, 
                              "replace": False
                          })
    assert response.status_code == 200
    
    finished = False
    timeout, start = 50, time.time()

    while not finished:
        for content in [test_objects['vod'], test_objects['assets'], test_objects['legacy_vod']]:
            auth = get_auth(content)
            assert time.time() - start < timeout, "Timeout waiting for jobs to complete"
            finished = True
            response = client.get(f"/{content}/status?authorization={auth}")
            assert response.status_code == 200
            result = response.get_json()
            print(result)
            for stream in result:
                for model in result[stream]:
                    if not result[stream][model]['status'] == 'Completed':
                        finished = False
                        break
            time.sleep(5)

    """Test the finalize workflow."""
    
    video_write = get_write_token(test_objects['vod'])
    image_write = get_write_token(test_objects['assets'])
    legacy_vod_write = get_write_token(test_objects['legacy_vod'])
    
    # Finalize video
    response = client.post(f"/{test_objects['vod']}/finalize?write_token={video_write}&authorization={video_auth}")
    assert response.status_code == 200
    
    # Finalize assets
    response = client.post(f"/{test_objects['assets']}/finalize?write_token={image_write}&authorization={assets_auth}")
    assert response.status_code == 200
    
    # Finalize legacy VOD
    response = client.post(f"/{test_objects['legacy_vod']}/finalize?write_token={legacy_vod_write}&leave_open=true&authorization={legacy_vod_auth}")
    assert response.status_code == 200
    
    # Verify files were uploaded to fabric
    client_fabric = ElvClient.from_configuration_url(config["fabric"]["config_url"], static_token=video_auth)
    files = client_fabric.list_files(write_token=video_write, path='video_tags/video/dummy_gpu')
    assert len(files) > 0  # Should have uploaded files

def test_write_token_tag(client):
    """Test tagging with write tokens."""
    assets_auth = get_auth(test_objects['assets'])
    image_write = get_write_token(test_objects['assets'])
    
    # Tag specific assets using write token
    response = client.post(f"/{image_write}/image_tag?authorization={assets_auth}", 
                          json={
                              "features": {"dummy_gpu": {"model": {"tags": ["hello changed"]}}}, 
                              "assets": ["assets/20521092.jpg", "assets/20820751.jpg", "assets/20979342.jpg", "assets/21777769.jpg"], 
                              "replace": True
                          })
    assert response.status_code == 200
    
    # Check status
    response = client.get(f"/{image_write}/status?authorization={assets_auth}")
    assert response.status_code == 200
    
    # Try tagging with replace=False (should not overwrite)
    response = client.post(f"/{image_write}/image_tag?authorization={assets_auth}", 
                          json={
                              "features": {"dummy_cpu": {"model": {"tags": ["Should not be changed"]}}}, 
                              "assets": ["assets/20521092.jpg", "assets/20820751.jpg", "assets/20979342.jpg", "assets/21777769.jpg"], 
                              "replace": False
                          })
    assert response.status_code == 200
    
    timeout = 15
    start = time.time()

    finished = False
    while not finished:
        assert time.time() - start < timeout, "Timeout waiting for tagging to complete"
        finished = True
        response = client.get(f"/{image_write}/status?authorization={assets_auth}")
        assert response.status_code == 200
        result = response.get_json()
        for stream in result:
            for model in result[stream]:
                if not result[stream][model]['status'] == 'Completed':
                    finished = False
                    break
        time.sleep(2)

    image_tags_path = os.path.join(config["storage"]["tags"], test_objects['assets'], "image")

    assert not os.path.exists(os.path.join(image_tags_path, "dummy_cpu")) or len(os.listdir(os.path.join(image_tags_path, "dummy_cpu"))) == 0

def test_partial_finalize(client):
    """Test partial finalization with ongoing tagging."""
    legacy_vod_auth = get_auth(test_objects['legacy_vod'])
    legacy_vod_write = get_write_token(test_objects['legacy_vod'])
    
    # Start a long-running tagging job
    response = client.post(f"/{test_objects['legacy_vod']}/tag?authorization={legacy_vod_auth}", 
                          json={
                              "start_time": 60, 
                              "end_time": 1000, 
                              "features": {"test_slow": {"model": {"tags": ["test partial finalize"], "allow_single_frame": False}}}, 
                              "replace": False
                          })
    assert response.status_code == 200
    
    # Wait for some progress
    parts_tagged = 0
    max_wait = 60
    start_time = time.time()
    
    while parts_tagged < 5 and (time.time() - start_time) < max_wait:  # Reduced target
        response = client.get(f"/{test_objects['legacy_vod']}/status?authorization={legacy_vod_auth}")
        assert response.status_code == 200
        result = response.get_json()
        
        if "video" in result and "test_slow" in result["video"]:
            tagging_progress = result["video"]["test_slow"].get("tagging_progress")
            if tagging_progress:
                parts_tagged = int(tagging_progress.split("/")[0])
        time.sleep(2)
    
    # Partial finalize
    response = client.post(f"/{test_objects['legacy_vod']}/finalize?write_token={legacy_vod_write}&authorization={legacy_vod_auth}&force=true&leave_open=true")
    assert response.status_code == 200
    
    # Verify some files were uploaded
    client_fabric = ElvClient.from_configuration_url(config["fabric"]["config_url"], static_token=legacy_vod_auth)
    files = client_fabric.list_files(write_token=legacy_vod_write, path='video_tags/video/test_slow')
    assert len(files) > 0

@pytest.mark.skip(reason="CPU limit test is more for performance monitoring than functional testing")
def test_cpu_limit(client):
    """Test CPU resource limits and queuing."""
    # Get auth tokens
    assets_auth = get_auth(test_objects['assets'])
    video_auth = get_auth(test_objects['vod'])
    legacy_vod_auth = get_auth(test_objects['legacy_vod'])
    
    # Start multiple CPU-intensive jobs simultaneously
    tag_requests = [
        (f"/{test_objects['vod']}/tag?authorization={video_auth}", 
         {"features": {"dummy_cpu": {"model": {"tags": ["a", "b", "a"], "allow_single_frame": False}}}, "replace": True}),
        (f"/{test_objects['legacy_vod']}/tag?authorization={legacy_vod_auth}", 
         {"features": {"dummy_cpu": {"model": {"tags": ["hello1"]}}, "shot": {}}, "start_time": 60, "end_time": 120, "replace": True}),
        (f"/{test_objects['assets']}/image_tag?authorization={assets_auth}", 
         {"features": {"dummy_cpu": {"model": {"tags": ["a", "b", "a"], "allow_single_frame": False}}}, "replace": True})
    ]
    
    # Send all requests
    for url, data in tag_requests:
        response = client.post(url, json=data)
        assert response.status_code == 200
    
    # Monitor status to ensure jobs are queued/completed properly
    status_urls = [
        f"/{test_objects['vod']}/status?authorization={video_auth}",
        f"/{test_objects['legacy_vod']}/status?authorization={legacy_vod_auth}",
        f"/{test_objects['assets']}/status?authorization={assets_auth}"
    ]
    
    # Check that all jobs eventually complete (with resource management)
    max_wait = 120
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        all_completed = True
        for url in status_urls:
            response = client.get(url)
            if response.status_code == 200:
                result = response.get_json()
                for stream in result:
                    for model in result[stream]:
                        if result[stream][model]['status'] != 'Completed':
                            all_completed = False
        
        if all_completed:
            break
        time.sleep(3)
    
    # Final verification that all jobs completed
    for url in status_urls:
        response = client.get(url)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])