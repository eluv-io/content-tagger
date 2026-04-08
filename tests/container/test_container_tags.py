import pytest
import json
import os
from unittest.mock import Mock, patch

from src.common.content import Content
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
def container_spec(temp_dir):
    output_path = os.path.join(temp_dir, "output", "output.jsonl")
    
    return ContainerSpec(
        id="test_video_container",
        media_dir=temp_dir,
        run_config={},
        logs_path=os.path.join(temp_dir, "logs"),
        cache_dir=os.path.join(temp_dir, "cache"),
        output_path=output_path,
        model_config=ModelConfig(
            type="video",
            description="Test model",
            image="test/model:latest",
            resources={}
        ),
        q=Content(qid="", token="")
    )


@pytest.fixture
def tag_container(mock_podman_client, container_spec):
    container = TagContainer(mock_podman_client, container_spec)

    class EchoDict(dict):
        def __getitem__(self, key):
            return key
    
    # this is so we can write tags directly to the output-path without first registering
    # the media files with add_media
    container.basename_to_source = EchoDict()

    return container


def write_jsonl(output_path: str, messages: list[dict]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def test_tags_basic(tag_container):
    output_path = tag_container.cfg.output_path
    
    write_jsonl(output_path, [
        {"type": "tag", "data": {"start_time": 0, "end_time": 5000, "tag": "person walking", "source_media": "video1.mp4"}},
        {"type": "tag", "data": {"start_time": 10000, "end_time": 15500, "tag": "car driving", "source_media": "video1.mp4"}},
    ])
    
    outputs = tag_container.tags()
    
    assert len(outputs) == 2
    
    tag1 = outputs[0]
    assert tag1.source_media.endswith("video1.mp4")
    assert tag1.start_time == 0
    assert tag1.end_time == 5000
    assert tag1.text == "person walking"
    
    tag2 = outputs[1]
    assert tag2.source_media.endswith("video1.mp4")
    assert tag2.start_time == 10000
    assert tag2.end_time == 15500
    assert tag2.text == "car driving"


def test_tags_with_frame_info(tag_container):
    output_path = tag_container.cfg.output_path
    
    write_jsonl(output_path, [
        {"type": "tag", "data": {
            "start_time": 1.0, "end_time": 3.5, "tag": "person walking", "source_media": "video1.mp4",
        }},
        {"type": "tag", "data": {
            "start_time": 1.0, "end_time": 1.0, "tag": "person walking", "source_media": "video1.mp4",
            "frame_info": {"frame_idx": 30, "box": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}},
            "additional_info": {"confidence": 0.95},
        }},
        {"type": "tag", "data": {
            "start_time": 3.0, "end_time": 3.0, "tag": "person walking", "source_media": "video1.mp4",
            "frame_info": {"frame_idx": 90, "box": {"x1": 0.11, "y1": 0.105, "x2": 0.21, "y2": 0.205}},
            "additional_info": {"confidence": 0.85},
        }},
    ])
    
    outputs = tag_container.tags()
    
    assert len(outputs) == 3
    
    video_tags = [t for t in outputs if t.frame_info is None]
    frame_tags = [t for t in outputs if t.frame_info is not None]
    
    assert len(video_tags) == 1
    assert video_tags[0].text == "person walking"
    
    assert len(frame_tags) == 2
    for ft in frame_tags:
        assert ft.text == "person walking"
        assert ft.start_time == ft.end_time
        assert ft.frame_info["box"] is not None
        assert ft.additional_info.get("confidence") is not None


def test_tags_multiple_sources(tag_container):
    output_path = tag_container.cfg.output_path
    
    write_jsonl(output_path, [
        {"type": "tag", "data": {"start_time": 0, "end_time": 5.0, "tag": "scene1", "source_media": "video1.mp4"}},
        {"type": "tag", "data": {"start_time": 0, "end_time": 3.0, "tag": "scene2", "source_media": "video2.mp4"}},
    ])
    
    outputs = tag_container.tags()
    
    assert len(outputs) == 2
    
    video1_tags = [tag for tag in outputs if tag.source_media.endswith("video1.mp4")]
    video2_tags = [tag for tag in outputs if tag.source_media.endswith("video2.mp4")]
    
    assert len(video1_tags) == 1
    assert video1_tags[0].text == "scene1"
    
    assert len(video2_tags) == 1
    assert video2_tags[0].text == "scene2"


def test_tags_image_files(tag_container):
    output_path = tag_container.cfg.output_path
    
    write_jsonl(output_path, [
        {"type": "tag", "data": {
            "start_time": 0, "end_time": 0, "tag": "cat", "source_media": "image1.jpg",
            "frame_info": {"frame_idx": 0, "box": {"x1": 0.05, "y1": 0.05, "x2": 0.15, "y2": 0.15}},
            "additional_info": {"confidence": 0.9},
        }},
        {"type": "tag", "data": {
            "start_time": 0, "end_time": 0, "tag": "dog", "source_media": "image1.jpg",
            "frame_info": {"frame_idx": 0, "box": {"x1": 0.2, "y1": 0.2, "x2": 0.3, "y2": 0.3}},
            "additional_info": {"confidence": 0.8},
        }},
    ])
    
    outputs = tag_container.tags()
    
    assert len(outputs) == 2
    
    tag1 = outputs[0]
    assert tag1.source_media.endswith("image1.jpg")
    assert tag1.start_time == 0
    assert tag1.end_time == 0
    assert tag1.text == "cat"
    assert tag1.frame_info["frame_idx"] == 0
    assert tag1.additional_info["confidence"] == 0.9
    
    tag2 = outputs[1]
    assert tag2.text == "dog"
    assert tag2.frame_info["frame_idx"] == 0
    assert tag2.additional_info["confidence"] == 0.8


def test_tags_no_output_file(tag_container):
    outputs = tag_container.tags()
    assert outputs == []


def test_tags_empty_output_file(tag_container):
    output_path = tag_container.cfg.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        pass
    outputs = tag_container.tags()
    assert outputs == []


def test_tags_incomplete_json_line_skipped(tag_container):
    output_path = tag_container.cfg.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(json.dumps({"type": "tag", "data": {"start_time": 0, "end_time": 5.0, "tag": "person walking", "source_media": "video1.mp4"}}) + "\n")
        f.write('{"type": "tag", "data": {"start_time": 10, "end_time": 15, "tag": "incomplete')  # incomplete
    
    outputs = tag_container.tags()
    assert len(outputs) == 1
    assert outputs[0].text == "person walking"


def test_errors_from_output(tag_container):
    output_path = tag_container.cfg.output_path
    
    write_jsonl(output_path, [
        {"type": "tag", "data": {"start_time": 0, "end_time": 5.0, "tag": "person walking", "source_media": "video1.mp4"}},
        {"type": "error", "data": {"message": "unsupported format", "source_media": "video2.mp4"}},
    ])
    
    tags = tag_container.tags()
    errors = tag_container.errors()
    
    assert len(tags) == 1
    assert len(errors) == 1
    assert errors[0].message == "unsupported format"
    assert errors[0].source_media == "video2.mp4"


def test_progress_from_output(tag_container):
    output_path = tag_container.cfg.output_path
    
    write_jsonl(output_path, [
        {"type": "tag", "data": {"start_time": 0, "end_time": 5.0, "tag": "person walking", "source_media": "video1.mp4"}},
        {"type": "progress", "data": {"source_media": "video1.mp4"}},
    ])
    
    tags = tag_container.tags()
    progress = tag_container.progress()
    
    assert len(tags) == 1
    assert len(progress) == 1
    assert progress[0].source_media == "video1.mp4"


def test_track_field(tag_container):
    output_path = tag_container.cfg.output_path
    
    write_jsonl(output_path, [
        {"type": "tag", "data": {"start_time": 0, "end_time": 5.0, "tag": "person walking", "track": "track1", "source_media": "video1.mp4"}},
        {"type": "tag", "data": {"start_time": 10.0, "end_time": 15.0, "tag": "car driving", "track": "track2", "source_media": "video1.mp4"}},
    ])
    
    tags = tag_container.tags()
    
    assert len(tags) == 2
    assert tags[0].model_track == "track1"
    assert tags[1].model_track == "track2"


def test_mixed_message_types(tag_container):
    output_path = tag_container.cfg.output_path
    
    write_jsonl(output_path, [
        {"type": "tag", "data": {"start_time": 0, "end_time": 5.0, "tag": "person walking", "source_media": "video1.mp4"}},
        {"type": "progress", "data": {"source_media": "media/video1.mp4"}},
        {"type": "tag", "data": {"start_time": 0, "end_time": 3.0, "tag": "car driving", "source_media": "video2.mp4"}},
        {"type": "error", "data": {"message": "something went wrong", "source_media": "video2.mp4"}},
        {"type": "progress", "data": {"source_media": "media/video2.mp4"}},
    ])
    
    tags = tag_container.tags()
    errors = tag_container.errors()
    progress = tag_container.progress()
    
    assert len(tags) == 2
    assert len(errors) == 1
    assert len(progress) == 2