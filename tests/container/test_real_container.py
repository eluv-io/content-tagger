
import pytest
import podman
import os
from loguru import logger
import time
import subprocess

from src.common.content import Content
from src.tag_containers.containers import TagContainer
from src.tag_containers.model import ContainerSpec, ModelConfig
from src.common.model import SystemResources

@pytest.fixture(autouse=True)
def check_skip():
    # check if dummy_gpu & dummy_cpu are available images in podman
    with podman.PodmanClient() as client:
        images = sum([image.tags for image in client.images.list() if image.tags], [])
        if "localhost/test_model:latest" not in images:
            logger.warning("Test model image 'localhost/test_model:latest' not found.")
            pytest.skip("Required test images not found in local podman registry")

def create_test_video(filepath: str, duration: float = 2.0):
    """Create a simple black video file using ffmpeg"""
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("ffmpeg not available")
    
    # Create a 1 second black video at 1 fps
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"color=c=black:s=320x240:d={duration}",
        "-r", "1",  # 1 fps
        "-pix_fmt", "yuv420p",
        "-y",  # overwrite
        filepath
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)

@pytest.fixture
def video_paths(temp_dir):
    videos_dir = os.path.join(temp_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    video1 = os.path.join(videos_dir, "video1.mp4")
    video2 = os.path.join(videos_dir, "video2.mp4")
    
    create_test_video(video1)
    create_test_video(video2)
    
    return video1, video2

@pytest.fixture
def container_spec(temp_dir):
    videos_dir = os.path.join(temp_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    output_path = os.path.join(temp_dir, "output", "output.jsonl")
    cache_dir = os.path.join(temp_dir, "cache")
    logs_path = os.path.join(temp_dir, "container.log")
    os.makedirs(cache_dir, exist_ok=True)

    return ContainerSpec(
        id="test_live_container",
        media_dir=videos_dir,
        run_config={},
        logs_path=logs_path,
        cache_dir=cache_dir,
        output_path=output_path,
        model_config=ModelConfig(
            type="video",
            description="Test model",
            image="localhost/test_model:latest",
            resources=SystemResources()
        ),
        q=Content(qid="q123", token="auth12345")
    )

@pytest.fixture
def container(container_spec):
    pclient = podman.PodmanClient()
    container = TagContainer(pclient, container_spec)

    yield container
    container.stop()

def test_container(container, video_paths):
    video1, video2 = video_paths
    
    try:
        # add some media before it starts
        container.add_media([video1])

        # Start the container
        container.start(gpuidx=None)
        
        # Wait for container to be ready
        time.sleep(1)
        assert container.is_running()

        # check that we have tags for the first video
        assert len(container.new_tags()) > 0
        assert container.errors() == []
        assert len(container.progress()) == 1
        assert container.progress()[0].source_media == video1
        
        # Create new video files
        video2 = os.path.join(container.cfg.media_dir, "video2.mp4")
        video3 = os.path.join(container.cfg.media_dir, "video3.mp4")
        create_test_video(video2)
        create_test_video(video3)
        
        # Add new media to running container
        container.add_media([video2, video3])
        
        # Wait for container to process
        time.sleep(1)

        statuses = container.progress()
        assert len(statuses) == 3
        assert statuses[0].source_media == video1
        assert statuses[1].source_media == video2
        assert statuses[2].source_media == video3

        # create a file with no media, should error
        bad_path = os.path.join(container.cfg.media_dir, "bad")
        open(bad_path, "a").close()

        # now add a bad file and check for error
        container.add_media([bad_path])

        time.sleep(1)

        assert len(container.progress()) == 3
        errors = container.errors()
        assert len(errors) == 1
        assert errors[0].message

        assert not container.is_running()
        
        # Check the container logs
        assert os.path.exists(container.cfg.logs_path)
        with open(container.cfg.logs_path, 'r') as f:
            logs = f.read()
        
        assert "Got media/video1.mp4" in logs
        assert "Got media/video2.mp4" in logs
        assert "Got media/video3.mp4" in logs

        # check that the model can read the auth environment variable
        assert "GOT AUTH: auth12345" in logs
        
    finally:
        # Cleanup
        if container.is_running():
            container.stop()
        
        # Verify container stopped
        assert not container.is_running()

def test_stop_container(container, video_paths):
    container.start(gpuidx=None)
    container.add_media(list(video_paths))
    time.sleep(1)
    assert container.is_running()
    container.stop()
    time.sleep(1)
    assert not container.is_running()

def test_stop_via_eof(container, video_paths):
    container.start(gpuidx=None)
    assert container.is_running()
    container.add_media(list(video_paths))
    container.send_eof()
    time.sleep(2)

    assert not container.is_running()
    # make sure it processed the files before eof'ing
    assert len(container.progress()) == 2

def test_generates_tags(container_spec, video_paths):
    container_spec.run_config = {"delay": 0.7}
    container = TagContainer(podman.PodmanClient(), container_spec)
    try:
        start = time.time()
        container.start(gpuidx=None)
        container.add_media(list(video_paths))
        assert container.is_running()
        counts = set()
        while time.time() - start < 10 and len(counts) < 3:
            counts.add(len(container.progress()))
        assert counts == {0, 1, 2}
    finally:
        if container.is_running():
            container.stop()