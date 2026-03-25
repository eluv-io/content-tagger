import os

import pytest
from unittest.mock import Mock, patch
import time
import json
from src.common.logging import logger
from src.common.content import Content

import podman
from src.api.arg_resolver import ArgsResolver
from src.fetch.model import DownloadRequest, FetchSession, VideoScope
from src.tagging.fabric_tagging.model import TagArgs, TagStartResult
from src.tagging.fabric_tagging.tagger import TaggerWorker
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.service.abstract import TaggerService
from tests.api.conftest import FakeLiveWorker

def is_queue_mode():
    return os.getenv("USE_QUEUE") == "true"

def is_job_success(client, q: Content):
    """Check if all jobs for content_id completed successfully."""
    response = client.get(f"/{q.qid}/job-status?authorization={q.token}")
    if response.status_code != 200:
        return False
    data = response.get_json()
    reports = data['jobs']
    return all(r['status'] == 'succeeded' for r in reports)

def wait_for_jobs_completion(client, contents: list[Content], timeout=30):
    """Wait for all jobs to complete."""
    start_time = time.time()
    if timeout is None:
        timeout = float('inf')
    while time.time() - start_time < timeout:
        all_finished = True
        
        for content in contents:
            response = client.get(f"/{content.qid}/job-status?authorization={content.token}")
            
            if response.status_code != 200:
                all_finished = False
                break

            data = response.get_json()
            reports = data['jobs']
            print(json.dumps(reports, indent=2))
            
            for report in reports:
                if report['status'] not in ['succeeded', 'failed', 'cancelled']:
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

def test_video_model(client, q):
    """Test the complete tagging workflow."""

    # Test initial status - should return 404 for no jobs
    response = client.get(f"/{q.qid}/job-status?authorization={q.token}")
    assert response.status_code == 404
    
    # Start video tagging with GPU feature
    response = client.post(
        f"/{q.qid}/tag?authorization={q.token}", 
        json={
            "options": {
                "destination_qid": "",
                "replace": True,
                "max_fetch_retries": 3,
                "scope": {}
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["hello1", "hello2"]},
                }
            ]
        }
    )
    assert response.status_code == 200
    completed = wait_for_jobs_completion(client, [q], timeout=30)
    assert completed
    tagstore: Tagstore = client.application.config["state"]["worker"].tagstore
    jobid = tagstore.find_batches(q=q)[0]
    tags = tagstore.find_tags(batch_id=jobid, q=q)
    tags = sorted(tags, key=lambda x: x.start_time)
    vtags = [t for t in tags if t.frame_info is None]
    # TODO: this randomly gave 124 one time and I can't reproduce it
    assert len(vtags) == 122
    next_tag = 'hello1'
    for tag in vtags:
        assert tag.text == next_tag
        next_tag = 'hello2' if next_tag == 'hello1' else 'hello1'

    ftags = [t for t in tags if t.frame_info is not None]
    assert len(ftags) == 122

    assert completed, "Timeout waiting for jobs to complete"

@pytest.mark.parametrize("last_res_has_media", [True, False])
def test_live_video_model(app, last_res_has_media, q):
    """Test the live tagging workflow with FakeLiveFetcher."""
    arg_resolver: ArgsResolver = app.config["state"]["arg_resolver"]
    arg_resolver.is_live_content = Mock(return_value=True)
    
    # Replace the real fetcher with FakeLiveFetcher
    tagger: TaggerWorker = app.config["state"]["worker"]
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


    tagger.fetcher.get_session = fake_get_worker
    
    # Create test client
    client = app.test_client()
    
    # Test initial status - should return 404 for no jobs
    response = client.get(f"/{q.qid}/job-status?authorization={q.token}")
    assert response.status_code == 404
    
    # Start video tagging with GPU feature
    response = client.post(
        f"/{q.qid}/tag?authorization={q.token}", 
        json={
            "options": {
                "replace": True,
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["hello1", "hello2"]},
                }
            ]
        }
    )
    assert response.status_code == 200
    
    # Wait for job completion
    completed = wait_for_jobs_completion(client, [q], timeout=60)
    assert completed, "Timeout waiting for live job to complete"

    assert is_job_success(client, q)
    
    # Calculate expected call count based on last_res_has_media
    expected_calls = 5 if last_res_has_media else 6
    
    # Verify the fetcher was called the expected number of times
    assert fake_worker_ref[0] is not None, "FakeLiveWorker was not created"
    assert fake_worker_ref[0].call_count == expected_calls, f"Expected {expected_calls} fetch calls, got {fake_worker_ref[0].call_count}"
    
    # Get tags and verify results
    jobid = tagstore.find_batches(q=q)[0]
    tags = tagstore.find_tags(batch_id=jobid, q=q)
    tags = sorted(tags, key=lambda x: x.start_time)

    vtags = [t for t in tags if t.frame_info is None]
    
    # Should have same number of tags as regular video model test
    assert len(vtags) == 122, f"Expected 122 tags, got {len(vtags)}"
    
    # Verify tags alternate between hello1 and hello2
    next_tag = 'hello1'
    for tag in vtags:
        assert tag.text == next_tag, f"Expected {next_tag}, got {tag.text}"
        next_tag = 'hello2' if next_tag == 'hello1' else 'hello1'
        
    logger.info(f"Live test completed successfully with {fake_worker_ref[0].call_count} fetch calls (last_res_has_media={last_res_has_media})")

def test_real_live_stream(app, q_live):
    """Test real live stream tagging with LiveWorker."""
    qid = q_live.qid
    auth = q_live.token
    
    tagger: TaggerWorker = app.config["state"]["worker"]
    tagstore = tagger.tagstore
    
    # Create test client
    client = app.test_client()
    
    # Start live tagging with test_model
    # Use small chunk_size and max_duration for faster testing
    response = client.post(
        f"/{qid}/tag?authorization={auth}", 
        json={
            "options": {
                "replace": True,
                "scope": {
                    "max_duration": 20,
                    "segment_length": 5
                }
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["hello1", "hello2"]},
                }
            ]
        }
    )
    assert response.status_code == 200
    
    # Wait for some segments to be processed (but not completion since it's live)
    # Check status periodically
    start_time = time.time()
    timeout = 25
    segments_found = False
    
    while time.time() - start_time < timeout:
        response = client.get(f"/{qid}/job-status?authorization={auth}")
        if response.status_code == 200:
            data = response.get_json()
            reports = data['jobs']
            print(json.dumps(reports, indent=2))
            
            # Find the test_model job in the list
            test_model_reports = [r for r in reports if r['model'] == 'test_model']
            if test_model_reports:
                report = test_model_reports[0]
                status = report['status']
                
                # Once we see completion, we know segments were processed
                if status == "succeeded":
                    segments_found = True
                    break
        
        time.sleep(3)
    
    # Verify final status is Stopped or Completed
    response = client.get(f"/{qid}/job-status?authorization={auth}")
    assert response.status_code == 200
    data = response.get_json()
    reports = data['jobs']
    
    test_model_report = next(r for r in reports if r['model'] == 'test_model')
    final_status = test_model_report['status']
    assert final_status in ['cancelled', 'succeeded'], f"Expected Stopped or Completed, got {final_status}"
    
    # verify we have some tags
    jobid = tagstore.find_batches(q=q_live, qid=q_live.qid)[0]
    tags = tagstore.find_tags(batch_id=jobid, q=q_live)
    tags = sorted(tags, key=lambda x: x.start_time)
    vtags = [ t for t in tags if t.frame_info is None and t.end_time > t.start_time]
    
    # Should have at least some tags from the segments
    assert len(vtags) >= 2

    last_wall_clock = 0
    for tag in vtags:
        assert tag.additional_info is not None
        assert 'timestamp_ms' in tag.additional_info
        assert tag.additional_info["timestamp_ms"] > last_wall_clock
        last_wall_clock = tag.additional_info["timestamp_ms"]

    logger.info(f"Live stream test completed successfully with {len(tags)} tags")

def test_asset_tag(client, q_assets):
    """Test asset tagging."""
    qid = q_assets.qid
    auth = q_assets.token
    
    # Start asset tagging with CPU feature
    response = client.post(
        f"/{qid}/tag?authorization={auth}", 
        json={
            "options": {
                "scope": {"type": "assets"}
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["hello world"]},
                }
            ]
        }
    )
    assert response.status_code == 200
    print('waiting')
    completed = wait_for_jobs_completion(client, [q_assets], timeout=25)
    print('tagging finished')
    assert completed
    status = client.get(f"/{qid}/job-status?authorization={auth}")
    print('got status')
    print(status.get_json())
    tagstore: FilesystemTagStore = client.application.config["state"]["worker"].tagstore
    jobid = tagstore.find_batches(q=q_assets)[0]
    print('got batches')
    tags = tagstore.find_tags(q=q_assets, batch_id=jobid)
    tags = sorted(tags, key=lambda x: x.start_time)
    assert len(tags) > 0

def test_stop_workflow(client, q):
    """Test stopping jobs."""
    video_auth = q.token
    
    # Start a job
    response = client.post(
        f"/{q.qid}/tag?authorization={video_auth}", 
        json={
            "options": {
                "replace": True,
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["test_stop"]},
                }
            ]
        }
    )
    assert response.status_code == 200
    
    # Stop the job quickly (before it completes)
    response = client.post(f"/{q.qid}/stop/test_model?authorization={video_auth}")
    assert response.status_code == 200
    assert len(response.get_json()["jobs"]) == 1

    time.sleep(1)
    
    # Check status - job should be stopped
    response = client.get(f"/{q.qid}/job-status?authorization={video_auth}")
    assert response.status_code == 200
    data = response.get_json()
    reports = data['jobs']
    
    # The job should exist and be in a stopped state
    test_model_reports = [r for r in reports if r['model'] == 'test_model']
    assert test_model_reports, "test_model job not found in status"
    status = test_model_reports[0]['status']
    assert status == 'cancelled', f"Expected job to be cancelled, got {status}"

def test_double_run(client, q):
    """Run same job twice, expect second to be rejected."""
    video_auth = q.token
    
    # Start initial job
    response = client.post(
        f"/{q.qid}/tag?authorization={video_auth}", 
        json={
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["original_tags"]},
                }
            ]
        }
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["jobs"][0]["started"] is True
    
    # Try to start another job with replace=False (should be rejected)
    response = client.post(
        f"/{q.qid}/tag?authorization={video_auth}", 
        json={

            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["should_not_replace"]},
                }
            ]
        }
    )
    data = response.get_json()
    assert response.status_code == 200
    assert data["jobs"][0]["started"] is False

    # stop the job
    start = time.time()
    response = client.post(f"/{q.qid}/stop/test_model?authorization={video_auth}")
    duration = time.time() - start

    assert duration < 2, f"Stop request took too long: {duration}s which is over 2s limit"
    assert response.status_code == 200



def test_find_default_audio_stream(q, app):
    resolver: ArgsResolver = app.config["state"]["arg_resolver"]
    result = resolver.find_default_audio_stream(q)

    assert result == "stereo"

def test_is_live_content(q_live, app):
    """Test the _is_live_content function."""
    resolver: ArgsResolver = app.config["state"]["arg_resolver"]
    assert resolver.is_live_content(q_live) == True

def test_stop_live_job(app, q_live):
    """Test that live jobs can be stopped cleanly mid-stream."""
    qid = q_live.qid
    auth = q_live.token
    
    worker: TaggerWorker = app.config["state"]["worker"]
    tagstore = worker.tagstore
    client = app.test_client()
    
    # Start live tagging with long duration
    response = client.post(
        f"/{qid}/tag?authorization={auth}", 
        json={
            "options": {
                "replace": True,
                "max_fetch_retries": 3,
                "scope": {
                    "max_duration": 60,
                    "segment_length": 4
                }
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["hello1", "hello2"]},
                }
            ]
        }
    )
    assert response.status_code == 200
    
    # Wait for job to start processing and get some tags
    start_time = time.time()
    some_tags_found = False
    
    while time.time() - start_time < 20:
        response = client.get(f"/{qid}/job-status?authorization={auth}")
        if response.status_code == 200:
            data = response.get_json()
            reports = data['jobs']
            test_model_reports = [r for r in reports if r['model'] == 'test_model']
            
            if test_model_reports:
                tag_details = test_model_reports[0]['tag_details']
                if tag_details is None:
                    progress = 0
                else:
                    progress = tag_details.get('progress', 0)
                # Wait until we have some progress but not complete
                if progress > 0:
                    # Check we actually have some tags
                    try:
                        jobid = tagstore.find_batches(q=q_live, qid=q_live.qid)[0]
                        tags = tagstore.find_tags(batch_id=jobid, q=q_live)
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
    response = client.get(f"/{qid}/job-status?authorization={auth}")
    data = response.get_json()
    reports = data['jobs']
    
    test_model_report = next(r for r in reports if r['model'] == 'test_model')
    final_status = test_model_report['status']
    assert final_status == 'cancelled', f"Expected cancelled, got {final_status}"

def test_invalid_model_name(client, q):
    """Test that requesting a non-existent model returns 400 error."""
    qid = q.qid
    auth = q.token

    # Try to tag with a model that doesn't exist
    response = client.post(
        f"/{qid}/tag?authorization={auth}", 
        json={

            "jobs": [
                {
                    "model": "nonexistent_model",
                    "model_params": {"tags": ["test"]},
                }
            ]
        }
    )
    
    assert response.status_code == 400

def test_stop_all_jobs(client, q):
    """Test stopping all jobs for a qid."""
    auth = q.token
    
    response = client.post(
        f"/{q.qid}/tag?authorization={auth}",
        json={
            "options": {
                "replace": True,
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["ok1", "ok2"]},
                },
                {
                    "model": "test_model2",
                    "model_params": {"tags": ["nope"]},
                }
            ]
        },
    )
    assert response.status_code == 200

    time.sleep(0.2)
    
    # Stop all jobs using the stop endpoint without feature specification
    stop_response = client.post(f"/{q.qid}/stop?authorization={auth}")
    assert stop_response.status_code == 200

    assert len(stop_response.get_json()["jobs"]) == 2
    
    # Wait a moment for stop to take effect
    time.sleep(1)
    
    # Check that all jobs are now stopped
    response = client.get(f"/{q.qid}/job-status?authorization={auth}")
    assert response.status_code == 200
    data = response.get_json()
    reports = data['jobs']
    
    for report in reports:
        assert report['status'] == 'cancelled', f"Expected cancelled, got {report['status']}"

def test_start_two_jobs_one_fails_partial_failure_response(client, q):
    """
    Start two jobs in one request; force tagger.tag() to raise for feature == 'fail_model'.
    Expect HTTP 200 with per-job start statuses (one success, one failure).
    """
    if is_queue_mode():
        pytest.skip()

    auth = q.token

    tagger: TaggerWorker = client.application.config["state"]["worker"]
    original_tag = tagger.tag

    def tag_wrapper(q: Content, args: TagArgs) -> TagStartResult:
        if args.feature == "test_model2":
            raise RuntimeError("boom")
        return original_tag(q, args)

    tagger.tag = tag_wrapper
    
    response = client.post(
        f"/{q.qid}/tag?authorization={auth}",
        json={
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["ok1", "ok2"]}
                },
                {
                    "model": "test_model2",
                    "model_params": {"tags": ["nope"]},
                }
            ]
        },
    )
    assert response.status_code == 200

    data = response.get_json()

    assert data["jobs"][0]["started"] is True
    assert data["jobs"][1]["started"] is False
    assert data["jobs"][1]["error"] == 'boom'

def test_status(client, q):
    """Test the job status endpoint with no jobs."""
    if not is_queue_mode():
        pytest.skip()

    response = client.get(f"/job-status?authorization={q.token}")
    assert response.status_code == 404

    response = client.post(
        f"/{q.qid}/tag?authorization={q.token}",
        json={
            "options": {
                "replace": True,
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["ok1", "ok2"]},
                },
                {
                    "model": "test_model2",
                    "model_params": {"tags": ["nope"]},
                }
            ]
        },
    )
    assert response.status_code == 200

    response = client.get(f"/job-status?authorization={q.token}")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["jobs"]) == 2

    response = client.get(f"/job-status?authorization={q.token}&tenant=iten2aYr2mCUKsJ6zL9e5kAXZ2mvDXom")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["jobs"]) == 2

    response = client.get(f"/job-status?authorization=invalid_token&tenant=another_tenant")
    assert response.status_code == 400

    response = client.get(f"/job-status?authorization={q.token}&tenant=another_tenant")
    assert response.status_code == 403