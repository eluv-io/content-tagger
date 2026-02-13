import pytest
import json
import os
from unittest.mock import Mock, patch

from src.tag_containers.containers import TagContainer
from src.tag_containers.model import ContainerSpec, ModelConfig
from src.common.model import SystemResources


@pytest.fixture
def mock_podman_client():
    return Mock()


@pytest.fixture
def video_files(temp_dir):
    video1 = os.path.join(temp_dir, "video1.mp4")
    video2 = os.path.join(temp_dir, "video2.mp4")
    
    open(video1, 'a').close()
    open(video2, 'a').close()
    
    return [video1, video2]


@pytest.fixture
def image_files(temp_dir):
    image1 = os.path.join(temp_dir, "image1.jpg")
    image2 = os.path.join(temp_dir, "image2.png")
    
    open(image1, 'a').close()
    open(image2, 'a').close()
    
    return [image1, image2]


@pytest.fixture
def container_spec_video(temp_dir, video_files):
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
    return TagContainer(mock_podman_client, container_spec_video)


@pytest.fixture
def image_tag_container(mock_podman_client, container_spec_image):
    return TagContainer(mock_podman_client, container_spec_image)


def create_video_tags_file(tags_dir: str, basename: str, tags_data: list[dict]):
    filename = f"{basename}_tags.json"
    filepath = os.path.join(tags_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(tags_data, f)
    return filename


def create_frame_tags_file(tags_dir: str, basename: str, frame_tags_data: dict):
    filename = f"{basename}_frametags.json"
    filepath = os.path.join(tags_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(frame_tags_data, f)
    return filename


def create_image_tags_file(tags_dir: str, basename: str, tags_data: list[dict]):
    filename = f"{basename}_imagetags.json"
    filepath = os.path.join(tags_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(tags_data, f)
    return filename


def test_tags_video_only_video_tags(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
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
        
        assert len(outputs) == 2
        
        tag1 = outputs[0]
        assert tag1.source_media.endswith("video1.mp4")
        assert tag1.start_time == 0
        assert tag1.end_time == 5
        assert tag1.text == "person walking"
        assert tag1.frame_tags == {}
        
        tag2 = outputs[1]
        assert tag2.source_media.endswith("video1.mp4")
        assert tag2.start_time == 10
        assert tag2.end_time == 15
        assert tag2.text == "car driving"


def test_tags_video_with_frame_tags(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 1000,
            "end_time": 3500,
            "text": "person walking",
            "confidence": 0.9
        }
    ]
    
    frame_tags_data = {
        30: [{
            "text": "person walking",
            "confidence": 0.95,
            "box": [0.1, 0.1, 0.2, 0.2]
        }],
        90: [{
            "text": "person walking", 
            "confidence": 0.85,
            "box": [0.11, 0.105, 0.21, 0.205]
        }],
        180: [{
            "text": "person walking",
            "confidence": 0.7,
            "box": [0.12, 0.11, 0.22, 0.21]
        }]
    }
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        
        tag = outputs[0]
        assert tag.text == "person walking"
        assert tag.frame_tags
        
        frame_info = tag.frame_tags
        assert len(frame_info) == 2
        
        assert "30" in frame_info
        assert "90" in frame_info
        assert frame_info["30"]["confidence"] == 0.95
        assert frame_info["30"]["box"] == [0.1, 0.1, 0.2, 0.2]
        assert frame_info["90"]["confidence"] == 0.85


def test_tags_video_multiple_sources(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video1_tags = [{"start_time": 0, "end_time": 5, "text": "scene1"}]
    create_video_tags_file(tags_dir, "video1.mp4", video1_tags)
    
    video2_tags = [{"start_time": 0, "end_time": 3, "text": "scene2"}]
    create_video_tags_file(tags_dir, "video2.mp4", video2_tags)

    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 2
        
        video1_tags = [tag for tag in outputs if tag.source_media.endswith("video1.mp4")]
        video2_tags = [tag for tag in outputs if tag.source_media.endswith("video2.mp4")]
        
        assert len(video1_tags) == 1
        assert video1_tags[0].text == "scene1"
        
        assert len(video2_tags) == 1
        assert video2_tags[0].text == "scene2"


def test_tags_image_files(image_tag_container):
    tags_dir = image_tag_container.cfg.tags_dir
    
    image_tags_data = [
        {
            "text": "cat",
            "confidence": 0.9,
            "box": [0.05, 0.05, 0.15, 0.15]
        },
        {
            "text": "dog",
            "confidence": 0.8,
            "box": [0.2, 0.2, 0.3, 0.3]
        }
    ]

    create_image_tags_file(tags_dir, "image1.jpg", image_tags_data)

    outputs = image_tag_container.tags()
    
    assert len(outputs) == 2
    
    tag1 = outputs[0]
    assert tag1.source_media.endswith("image1.jpg")
    assert tag1.start_time == 0
    assert tag1.end_time == 0
    assert tag1.text == "cat"
    assert tag1.frame_tags["0"]["confidence"] == 0.9
    assert tag1.frame_tags["0"]["box"] == [0.05, 0.05, 0.15, 0.15]
    
    tag2 = outputs[1]
    assert tag2.text == "dog"
    assert tag2.frame_tags["0"]["confidence"] == 0.8


def test_tags_no_files(video_tag_container):
    outputs = video_tag_container.tags()
    assert outputs == []


def test_tags_frame_tags_no_text_match(video_tag_container):
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
            "text": "car driving",
            "confidence": 0.9,
            "box": [0.1, 0.1, 0.2, 0.2]
        }
    }

    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)

    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        tag = outputs[0]
        assert tag.text == "person walking"
        assert not tag.frame_tags or len(tag.frame_tags) == 0


def test_tags_missing_video_tags_file(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    frame_tags_data = {30: {"text": "test", "confidence": 0.9}}
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        assert isinstance(outputs, list)


def test_source_from_filename(video_tag_container: TagContainer, image_tag_container: TagContainer):
    source = video_tag_container._source_from_filename("video1.mp4_tags.json")
    assert source.endswith("video1.mp4")
    
    source = video_tag_container._source_from_filename("video1.mp4_frametags.json") 
    assert source.endswith("video1.mp4")
    
    source = image_tag_container._source_from_filename("image1.jpg_imagetags.json")
    assert source.endswith("image1.jpg")


def test_tags_frame_tags_only_overlapping_frames(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 2,
            "end_time": 4,
            "text": "person walking"
        }
    ]
    
    frame_tags_data = {
        30: {
            "text": "person walking",
            "confidence": 0.9
        },
        60: {
            "text": "person walking",
            "confidence": 0.9
        },
        90: {
            "text": "person walking",
            "confidence": 0.9
        },
        120: {
            "text": "person walking",
            "confidence": 0.9
        },
        150: {
            "text": "person walking",
            "confidence": 0.9
        }
    }
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        tag = outputs[0]
        
        if tag.frame_tags:
            frame_tags = tag.frame_tags
            frame_indices = [int(fidx) for fidx in frame_tags]
            assert 30 not in frame_indices
            assert 150 not in frame_indices
            assert 60 in frame_indices
            assert 90 in frame_indices
            assert 120 in frame_indices


def test_tags_case_insensitive_text_matching(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5,
            "text": "Person Walking"
        }
    ]
    
    frame_tags_data = {
        30: {
            "text": "person walking",
            "confidence": 0.9
        },
        60: {
            "text": "PERSON WALKING",
            "confidence": 0.8
        }
    }
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container.tags()
        
        assert len(outputs) == 1
        tag = outputs[0]
        
        if tag.frame_tags:
            frame_tags = tag.frame_tags
            assert len(frame_tags) == 2

def test_container_with_directory_input(mock_podman_client, temp_dir):
    videos_dir = os.path.join(temp_dir, "videos")
    os.makedirs(videos_dir)
    
    video1 = os.path.join(videos_dir, "video1.mp4")
    video2 = os.path.join(videos_dir, "video2.mp4")
    open(video1, 'a').close()
    open(video2, 'a').close()
    
    tags_dir = os.path.join(temp_dir, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    
    container_spec = ContainerSpec(
        id="test_dir_container",
        media_input=videos_dir,
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
    
    video1_tags = [{"start_time": 0, "end_time": 5, "text": "scene1"}]
    create_video_tags_file(tags_dir, "video1.mp4", video1_tags)
    
    video2_tags = [{"start_time": 0, "end_time": 3, "text": "scene2"}]
    create_video_tags_file(tags_dir, "video2.mp4", video2_tags)

    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = container.tags()
        
        assert len(outputs) == 2
        
        video1_tags = [tag for tag in outputs if tag.source_media.endswith("video1.mp4")]
        video2_tags = [tag for tag in outputs if tag.source_media.endswith("video2.mp4")]
        
        assert len(video1_tags) == 1
        assert video1_tags[0].text == "scene1"
        
        assert len(video2_tags) == 1
        assert video2_tags[0].text == "scene2"

def test_tags_get_fps_only_called_with_frame_tags(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
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
        
        mock_get_fps.assert_not_called()
        
        assert len(outputs) == 1

def test_tags_get_fps_cached_per_video(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
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
            "box": [0.1, 0.1, 0.2, 0.2]
        }],
        "90": [{
            "text": "car driving",
            "confidence": 0.85,
            "box": [0.11, 0.11, 0.21, 0.21]
        }],
        "150": [{
            "text": "bird flying",
            "confidence": 0.8,
            "box": [0.12, 0.12, 0.22, 0.22]
        }]
    }
    
    create_video_tags_file(tags_dir, "video1.mp4", video_tags_data)
    create_frame_tags_file(tags_dir, "video1.mp4", frame_tags_data)
    
    with patch('src.tag_containers.containers.get_fps') as mock_get_fps:
        mock_get_fps.return_value = 30.0
        
        outputs = video_tag_container.tags()
        video_tag_container.tags()
        
        assert mock_get_fps.call_count == 1
        
        call_args = mock_get_fps.call_args[0]
        assert call_args[0].endswith("video1.mp4")
        
        assert len(outputs) == 3
        
        for tag in outputs:
            assert tag.frame_tags

def test_source_from_tag_file_with_source_media_field(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5,
            "text": "person walking",
            "source_media": "video1.mp4"
        },
        {
            "start_time": 10,
            "end_time": 15,
            "text": "car driving",
            "source_media": "video1.mp4"
        }
    ]
    
    tag_file = os.path.join(tags_dir, "arbitrary_name_tags.json")
    with open(tag_file, 'w') as f:
        json.dump(video_tags_data, f)
    
    source = video_tag_container._source_from_tag_file(tag_file)
    assert source.endswith("video1.mp4")

def test_source_from_tag_file_fallback_to_filename(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5,
            "text": "person walking"
        }
    ]
    
    tag_file = os.path.join(tags_dir, "video1.mp4_tags.json")
    with open(tag_file, 'w') as f:
        json.dump(video_tags_data, f)
    
    source = video_tag_container._source_from_tag_file(tag_file)
    assert source.endswith("video1.mp4")

def test_source_from_tag_file_empty_list(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    tag_file = os.path.join(tags_dir, "video1.mp4_tags.json")
    with open(tag_file, 'w') as f:
        json.dump([], f)
    
    source = video_tag_container._source_from_tag_file(tag_file)
    assert source.endswith("video1.mp4")

def test_source_from_tag_file_multiple_sources_fallback(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5,
            "text": "person walking",
            "source_media": "video1.mp4"
        },
        {
            "start_time": 10,
            "end_time": 15,
            "text": "car driving",
            "source_media": "video2.mp4"
        }
    ]
    
    tag_file = os.path.join(tags_dir, "video1.mp4_tags.json")
    with open(tag_file, 'w') as f:
        json.dump(video_tags_data, f)
    
    source = video_tag_container._source_from_tag_file(tag_file)
    assert source.endswith("video1.mp4")

def test_output_from_tags_video_only(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5000,
            "text": "person walking"
        },
        {
            "start_time": 10000,
            "end_time": 15000,
            "text": "car driving"
        }
    ]
    
    tag_file = os.path.join(tags_dir, "video1.mp4_tags.json")
    with open(tag_file, 'w') as f:
        json.dump(video_tags_data, f)
    
    with patch('src.tag_containers.containers.get_fps', return_value=30.0):
        outputs = video_tag_container._output_from_tags("video1.mp4", [tag_file])
    
    assert len(outputs) == 2
    assert outputs[0].start_time == 0
    assert outputs[0].end_time == 5000
    assert outputs[0].text == "person walking"
    assert outputs[0].source_media == "video1.mp4"
    
    assert outputs[1].start_time == 10000
    assert outputs[1].end_time == 15000
    assert outputs[1].text == "car driving"
    assert outputs[1].source_media == "video1.mp4"

def test_source_from_tags(video_tag_container):
    # test that we can resolve the source name from the tags if it's not in the filename
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5000,
            "text": "person walking",
            "source_media": "video1.mp4"
        },
        {
            "start_time": 10000,
            "end_time": 15000,
            "text": "car driving",
            "source_media": "video1.mp4"
        }
    ]
    
    tag_file = os.path.join(tags_dir, "asdf_tags.json")
    with open(tag_file, 'w') as f:
        json.dump(video_tags_data, f)
    
    outputs = video_tag_container.tags()
    
    assert len(outputs) == 2
    assert outputs[0].source_media.endswith("video1.mp4")
    assert outputs[1].source_media.endswith("video1.mp4")

def test_bad_tag_json(video_tag_container):
    """Test that incomplete/invalid JSON files are handled gracefully"""
    tags_dir = video_tag_container.cfg.tags_dir
    
    # Create a valid tag file
    valid_tags = [
        {
            "start_time": 0,
            "end_time": 5000,
            "text": "person walking",
            "source_media": "video1.mp4"
        }
    ]
    
    valid_file = os.path.join(tags_dir, "video1.mp4_tags.json")
    with open(valid_file, 'w') as f:
        json.dump(valid_tags, f)
    
    # Create an invalid/incomplete JSON file (simulates container still writing)
    invalid_file = os.path.join(tags_dir, "video2.mp4_tags.json")
    with open(invalid_file, 'w') as f:
        f.write('{"start_time": 0, "end_time": 5000, "text": "incomplete')  # Incomplete JSON
    
    # Should only return tags from valid file, skip invalid one
    outputs = video_tag_container.tags()
    
    assert len(outputs) == 1
    assert outputs[0].source_media.endswith("video1.mp4")
    assert outputs[0].text == "person walking"

def test_bad_filename(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5000,
            "text": "person walking",
            "source_media": "video1.mp4"
        },
        {
            "start_time": 10000,
            "end_time": 15000,
            "text": "car driving",
            "source_media": "video1.mp4"
        }
    ]
    
    tag_file = os.path.join(tags_dir, "asdf.json")
    with open(tag_file, 'w') as f:
        json.dump(video_tags_data, f)
    
    outputs = video_tag_container.tags()
    
    assert len(outputs) == 0

def test_track_field(video_tag_container):
    tags_dir = video_tag_container.cfg.tags_dir
    
    video_tags_data = [
        {
            "start_time": 0,
            "end_time": 5000,
            "text": "person walking",
            "track": "track1"
        },
        {
            "start_time": 10000,
            "end_time": 15000,
            "text": "car driving",
            "track": "track2"
        }
    ]
    
    tag_file = os.path.join(tags_dir, "video1.mp4_tags.json")
    with open(tag_file, 'w') as f:
        json.dump(video_tags_data, f)

    tags = video_tag_container.tags()

    assert len(tags) == 2
    assert tags[0].track == "track1"
    assert tags[1].track == "track2"