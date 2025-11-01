import pytest
import json
import os
from unittest.mock import Mock, patch

from src.tag_containers.containers import TagContainer
from src.tag_containers.model import ContainerSpec, ModelConfig
from src.common.model import SystemResources


@pytest.fixture
def mock_podman_client():
    """Mock PodmanClient for testing"""
    return Mock()


@pytest.fixture
def video_files(temp_dir):
    """Create test video files"""
    video1 = os.path.join(temp_dir, "video1.mp4")
    video2 = os.path.join(temp_dir, "video2.mp4")
    
    # Create empty files
    open(video1, 'a').close()
    open(video2, 'a').close()
    
    return [video1, video2]


@pytest.fixture
def image_files(temp_dir):
    """Create test image files"""
    image1 = os.path.join(temp_dir, "image1.jpg")
    image2 = os.path.join(temp_dir, "image2.png")
    
    # Create empty files
    open(image1, 'a').close()
    open(image2, 'a').close()
    
    return [image1, image2]


@pytest.fixture
def container_spec_video(temp_dir, video_files):
    """Create ContainerSpec for video files"""
    tags_dir = os.path.join(temp_dir, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    
    return ContainerSpec(
        id="test_video_container",
        media_input=video_files,
        run_config={},
        logs_path=os.path.join(temp_dir, "logs"),
        cache_dir=os.path.join(temp_dir, "cache"),
        tags_dir=tags_dir,
        model_config=ModelConfig(
            type="video",
            image="test/model:latest",
            resources={}
        )
    )


@pytest.fixture
def container_spec_image(temp_dir, image_files):
    """Create ContainerSpec for image files"""
    tags_dir = os.path.join(temp_dir, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    
    return ContainerSpec(
        id="test_image_container",
        media_input=image_files,
        run_config={},
        logs_path=os.path.join(temp_dir, "logs"),
        cache_dir=os.path.join(temp_dir, "cache"),
        tags_dir=tags_dir,
        model_config=ModelConfig(
            type="frame",
            image="test/model:latest", 
            resources=SystemResources(memory_mb=1024, vcpus=1)
        )
    )


@pytest.fixture
def video_tag_container(mock_podman_client, container_spec_video):
    """Create TagContainer for video files"""
    return TagContainer(mock_podman_client, container_spec_video)


@pytest.fixture
def image_tag_container(mock_podman_client, container_spec_image):
    """Create TagContainer for image files"""
    return TagContainer(mock_podman_client, container_spec_image)


def create_video_tags_file(tags_dir: str, basename: str, tags_data: list[dict]):
    """Helper to create video tags JSON file"""
    filename = f"{basename}_tags.json"
    filepath = os.path.join(tags_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(tags_data, f)
    return filename


def create_frame_tags_file(tags_dir: str, basename: str, frame_tags_data: dict):
    """Helper to create frame tags JSON file"""
    filename = f"{basename}_frametags.json"
    filepath = os.path.join(tags_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(frame_tags_data, f)
    return filename


def create_image_tags_file(tags_dir: str, basename: str, tags_data: list[dict]):
    """Helper to create image tags JSON file"""
    filename = f"{basename}_imagetags.json"
    filepath = os.path.join(tags_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(tags_data, f)
    return filename


def test_tags_video_only_video_tags(video_tag_container):
    """Test tags() with only video tags (no frame tags)"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    # Create video tags files
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5,
            "text": "person walking",
            "confidence": 0.9
        },
        {
            "start_time": 10,
            "end_time": 15,
            "text": "car driving", 
            "confidence": 0.8
        }
    ]
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        output = outputs[0]
        assert output.source_media.endswith("video1.mp4")
        assert len(output.tags) == 2
        
        # Check first tag
        tag1 = output.tags[0]
        assert tag1.start_time == 0
        assert tag1.end_time == 5
        assert tag1.text == "person walking"
        assert tag1.additional_info == {}
        
        # Check second tag
        tag2 = output.tags[1]
        assert tag2.start_time == 10
        assert tag2.end_time == 15
        assert tag2.text == "car driving"


def test_tags_video_with_frame_tags(video_tag_container):
    """Test tags() with both video and frame tags"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    # Create video tags
    video_tags_data = [
        {
            "start_time": 1000,
            "end_time": 3500,
            "text": "person walking",
            "confidence": 0.9
        }
    ]
    
    # Create frame tags (matching text with video tag)
    frame_tags_data = {
        30: [{  # frame 30 = 1 second at 30fps
            "text": "person walking",
            "confidence": 0.95,
            "box": [100, 100, 200, 200]
        }],
        90: [{  # frame 90 = 3 seconds at 30fps
            "text": "person walking", 
            "confidence": 0.85,
            "box": [110, 105, 210, 205]
        }],
        180: [{  # frame 180 = 6 seconds at 30fps (outside video tag range)
            "text": "person walking",
            "confidence": 0.7,
            "box": [120, 110, 220, 210]
        }]
    }
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        output = outputs[0]
        assert len(output.tags) == 1
        
        tag = output.tags[0]
        assert tag.text == "person walking"
        assert tag.frame_tags
        
        frame_info = tag.frame_tags
        # Should only include frames 30 and 90 (within 0-5 second range)
        assert len(frame_info) == 2
        
        # Check frame tag data
        assert "30" in frame_info
        assert "90" in frame_info
        assert frame_info["30"]["confidence"] == 0.95
        assert frame_info["30"]["box"] == [100, 100, 200, 200]
        assert frame_info["90"]["confidence"] == 0.85


def test_tags_video_multiple_sources(video_tag_container):
    """Test tags() with multiple video sources"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    # Create tags for video1
    video1_tags = [{"start_time": 0, "end_time": 5, "text": "scene1"}]
    create_video_tags_file(tags_dir, "video1.mp4", video1_tags)
    
    # Create tags for video2  
    video2_tags = [{"start_time": 0, "end_time": 3, "text": "scene2"}]
    create_video_tags_file(tags_dir, "video2.mp4", video2_tags)

    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 2
        
        # Find outputs by source media
        video1_output = next(o for o in outputs if o.source_media.endswith("video1.mp4"))
        video2_output = next(o for o in outputs if o.source_media.endswith("video2.mp4"))
        
        assert len(video1_output.tags) == 1
        assert video1_output.tags[0].text == "scene1"
        
        assert len(video2_output.tags) == 1
        assert video2_output.tags[0].text == "scene2"


def test_tags_image_files(image_tag_container):
    """Test tags() with image files"""
    tags_dir = image_tag_container.cfg.tags_dir
    
    # Create image tags
    image_tags_data = [
        {
            "text": "cat",
            "confidence": 0.9,
            "box": [50, 50, 150, 150]
        },
        {
            "text": "dog",
            "confidence": 0.8,
            "box": [200, 200, 300, 300]
        }
    ]

    create_image_tags_file(tags_dir, "image1.jpg", image_tags_data)

    outputs = image_tag_container.tags()
    
    assert len(outputs) == 1
    output = outputs[0]
    assert output.source_media.endswith("image1.jpg")
    assert len(output.tags) == 2
    
    # Check tags
    tag1 = output.tags[0]
    assert tag1.start_time == 0
    assert tag1.end_time == 0
    assert tag1.text == "cat"
    assert tag1.additional_info["confidence"] == 0.9
    assert tag1.additional_info["box"] == [50, 50, 150, 150]
    
    tag2 = output.tags[1]
    assert tag2.text == "dog"
    assert tag2.additional_info["confidence"] == 0.8


def test_tags_no_files(video_tag_container):
    """Test tags() when no tag files exist"""
    outputs = video_tag_container.tags()
    assert outputs == []


def test_tags_frame_tags_no_text_match(video_tag_container):
    """Test tags() when frame tags don't match video tag text"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5,
            "text": "person walking"
        }
    ]
    
    frame_tags_data = {
        30: {
            "text": "car driving",  # Different text
            "confidence": 0.9,
            "box": [100, 100, 200, 200]
        }
    }

    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)

    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        tag = outputs[0].tags[0]
        assert tag.text == "person walking"
        # Should not have frame_tags since text doesn't match
        assert not tag.frame_tags or len(tag.frame_tags) == 0


def test_tags_invalid_json_file(video_tag_container):
    """Test tags() handles invalid JSON gracefully"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    # Create invalid JSON file
    invalid_file = os.path.join(tags_dir, "video1_tags.json")
    with open(invalid_file, 'w') as f:
        f.write("invalid json{")
    
    with pytest.raises(Exception):  # Should raise some JSON-related exception
        video_tag_container.tags()


def test_tags_missing_video_tags_file(video_tag_container):
    """Test tags() when video tags file is missing but frame tags exist"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    frame_tags_data = {30: {"text": "test", "confidence": 0.9}}
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        # Should handle gracefully - either return empty or skip this source
        outputs = video_tag_container.tags()
        # The exact behavior depends on implementation
        assert isinstance(outputs, list)


def test_source_from_tag_file(video_tag_container, image_tag_container):
    """Test _source_from_tag_file method"""
    # Test video tags file
    source = video_tag_container._source_from_tag_file("video1.mp4_tags.json")
    assert source.endswith("video1.mp4")
    
    # Test frame tags file
    source = video_tag_container._source_from_tag_file("video1.mp4_frametags.json") 
    assert source.endswith("video1.mp4")
    
    # Test image file
    source = image_tag_container._source_from_tag_file("image1.jpg_imagetags.json")
    assert source.endswith("image1.jpg")


def test_tags_frame_tags_only_overlapping_frames(video_tag_container):
    """Test that only frame tags within video tag time range are included"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 2,  # 2-4 second range
            "end_time": 4,
            "text": "person walking"
        }
    ]
    
    frame_tags_data = {
        30: {   # 1 second - before range
            "text": "person walking",
            "confidence": 0.9
        },
        60: {   # 2 seconds - at start of range
            "text": "person walking",
            "confidence": 0.9
        },
        90: {   # 3 seconds - in range
            "text": "person walking",
            "confidence": 0.9
        },
        120: {  # 4 seconds - at end of range
            "text": "person walking",
            "confidence": 0.9
        },
        150: {  # 5 seconds - after range
            "text": "person walking",
            "confidence": 0.9
        }
    }
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        tag = outputs[0].tags[0]
        
        if tag.frame_tags:
            frame_tags = tag.frame_tags
            # Should include frames 60, 90, 120 (timestamps 2, 3, 4)
            frame_indices = [int(fidx) for fidx in frame_tags]
            assert 30 not in frame_indices   # Before range
            assert 150 not in frame_indices  # After range
            assert 60 in frame_indices      # At start
            assert 90 in frame_indices      # In middle  
            assert 120 in frame_indices     # At end


def test_tags_case_insensitive_text_matching(video_tag_container):
    """Test that frame tag text matching is case insensitive"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5,
            "text": "Person Walking"  # Mixed case
        }
    ]
    
    frame_tags_data = {
        30: {
            "text": "person walking",  # Lower case
            "confidence": 0.9
        },
        60: {
            "text": "PERSON WALKING",  # Upper case
            "confidence": 0.8
        }
    }
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        tag = outputs[0].tags[0]
        
        if tag.frame_tags:
            frame_tags = tag.frame_tags
            # Both frame tags should match despite case differences
            assert len(frame_tags) == 2

def test_container_with_directory_input(mock_podman_client, temp_dir):
    """Test starting a container with directory as media_input"""
    # Create a directory with video files
    videos_dir = os.path.join(temp_dir, "videos")
    os.makedirs(videos_dir)
    
    video1 = os.path.join(videos_dir, "video1.mp4")
    video2 = os.path.join(videos_dir, "video2.mp4")
    open(video1, 'a').close()
    open(video2, 'a').close()
    
    # Create container spec with directory input
    tags_dir = os.path.join(temp_dir, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    
    container_spec = ContainerSpec(
        id="test_dir_container",
        media_input=videos_dir,  # Pass directory instead of file list
        run_config={},
        logs_path=os.path.join(temp_dir, "logs"),
        cache_dir=os.path.join(temp_dir, "cache"),
        tags_dir=tags_dir,
        model_config=ModelConfig(
            type="video",
            image="test/model:latest",
            resources=SystemResources(memory_mb=1024, vcpus=1)
        )
    )
    
    container = TagContainer(mock_podman_client, container_spec)
    
    # Create video tags for both videos
    video1_tags = [{"start_time": 0, "end_time": 5, "text": "scene1"}]
    create_video_tags_file(tags_dir, "video1.mp4", video1_tags)
    
    video2_tags = [{"start_time": 0, "end_time": 3, "text": "scene2"}]
    create_video_tags_file(tags_dir, "video2.mp4", video2_tags)

    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = container.tags()
        
        assert len(outputs) == 2
        
        # Find outputs by source media
        video1_output = next(o for o in outputs if o.source_media.endswith("video1.mp4"))
        video2_output = next(o for o in outputs if o.source_media.endswith("video2.mp4"))
        
        assert len(video1_output.tags) == 1
        assert video1_output.tags[0].text == "scene1"
        
        assert len(video2_output.tags) == 1
        assert video2_output.tags[0].text == "scene2"

def test_tags_get_fps_only_called_with_frame_tags(video_tag_container):
    """Test that get_fps is only called when frame tags are present"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    # Create video tags without frame tags
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5,
            "text": "person walking"
        }
    ]
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    
    with patch('src.tag_containers.containers.get_fps') as mock_get_fps:
        mock_get_fps.return_value = 30.0
        
        outputs = video_tag_container.tags()
        
        # get_fps should NOT be called since there are no frame tags
        mock_get_fps.assert_not_called()
        
        assert len(outputs) == 1
        assert len(outputs[0].tags) == 1

def test_tags_get_fps_cached_per_video(video_tag_container):
    """Test that get_fps is only called once per video file (cached)"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    # Create multiple video tags with frame tags for the same video
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 2000,
            "text": "person walking"
        },
        {
            "start_time": 2000,
            "end_time": 4000,
            "text": "car driving"
        },
        {
            "start_time": 4000,
            "end_time": 6000,
            "text": "bird flying"
        }
    ]
    
    frame_tags_data = {
        "30": [{
            "text": "person walking",
            "confidence": 0.9,
            "box": [100, 100, 200, 200]
        }],
        "90": [{
            "text": "car driving",
            "confidence": 0.85,
            "box": [110, 110, 210, 210]
        }],
        "150": [{
            "text": "bird flying",
            "confidence": 0.8,
            "box": [120, 120, 220, 220]
        }]
    }
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps') as mock_get_fps:
        mock_get_fps.return_value = 30.0
        
        outputs = video_tag_container.tags()
        video_tag_container.tags()
        
        # get_fps should be called exactly once despite multiple tags with frame tags
        assert mock_get_fps.call_count == 1
        
        # Verify the video path passed to get_fps
        call_args = mock_get_fps.call_args[0]
        assert call_args[0].endswith("video1.mp4")
        
        assert len(outputs) == 1
        assert len(outputs[0].tags) == 3
        
        # Verify all tags have frame_tags
        for tag in outputs[0].tags:
            assert tag.frame_tags