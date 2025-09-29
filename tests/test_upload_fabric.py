import pytest
import tempfile
import shutil
import os
import json
from dotenv import load_dotenv

from src.tags.conversion_workflow import upload_tags_to_fabric
from src.tags.conversion import TagConverter, TagConverterConfig
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.tags.tagstore.rest_tagstore import RestTagstore
from src.tags.tagstore.types import Tag
from src.common.content import Content, ContentConfig, ContentFactory

load_dotenv()

TEST_QHIT = "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm"

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def rest_tagstore() -> RestTagstore:
    """Create a RestTagstore using TEST_TAGSTORE_HOST environment variable"""
    host = os.getenv("TEST_TAGSTORE_HOST")
    
    return RestTagstore(base_url=f"http://{host}")

@pytest.fixture
def filesystem_tagstore(temp_dir: str) -> FilesystemTagStore:
    """Create a FilesystemTagStore with test data"""
    store = FilesystemTagStore(base_dir=temp_dir)
    return store

@pytest.fixture
def tagstore(rest_tagstore: RestTagstore, filesystem_tagstore: FilesystemTagStore):
    """Create appropriate tagstore based on TEST_TAGSTORE_HOST environment variable"""
    if os.getenv("TEST_TAGSTORE_HOST"):
        return rest_tagstore
    else:
        return filesystem_tagstore

@pytest.fixture
def tag_converter() -> TagConverter:
    """Create a TagConverter with test configuration"""
    config = TagConverterConfig(
        interval=5,  # 5-minute buckets for testing
        name_mapping={
            "object": "Object Detection",
            "asr": "Speech to Text",
            "shot": "Shot Detection"
        },
        single_tag_tracks=[],
        coalesce_tracks=["asr"],
        max_sentence_words=10
    )
    return TagConverter(config)

@pytest.fixture
def q():
    """Create Content object with write token from environment"""
    auth_token = os.getenv("TEST_AUTH")
    write_token = os.getenv("TEST_QWT")
    
    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")
    if not write_token:
        pytest.skip("TEST_QWT not set in environment")
    
    cfg = ContentConfig(
        config_url="https://host-154-14-185-98.contentfabric.io/config?self&qspace=main", 
        parts_url="https://host-154-14-185-98.contentfabric.io/config?self&qspace=main"
    )
    factory = ContentFactory(cfg=cfg)
    
    q = factory.create_content(qhit=write_token, auth=auth_token)

    yield q
    q.replace_metadata(metadata_subtree="video_tags", metadata={})

def test_upload_tags_to_fabric_full_workflow(
    tagstore: Tagstore,
    tag_converter: TagConverter,
    q: Content,
    temp_dir: str,
):
    """Test complete upload workflow with multiple tracks and time buckets"""

    qhit = q.qhit

    # Create test jobs in tagstore
    obj_job = tagstore.start_job(qhit, track="object", author="tagger", stream="video")
    asr_job = tagstore.start_job(qhit, stream="audio", author="tagger", track="asr")
    shot_job = tagstore.start_job(qhit, stream="video", author="tagger", track="shot")

    # Create tags spanning multiple time buckets (5-minute intervals = 300,000ms)
    # First bucket: 0-300,000ms
    obj_tags_bucket1 = [
        Tag(0, 5000, "person", {
            "frame_tags": {
                "1000": {"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.9},
                "3000": {"box": {"x1": 15, "y1": 25, "x2": 35, "y2": 45}, "confidence": 0.8}
            }
        }, "part_0.mp4", obj_job.id),
        Tag(10000, 15000, "car", {
            "frame_tags": {
                "12000": {"box": {"x1": 50, "y1": 60, "x2": 70, "y2": 80}, "confidence": 0.95}
            }
        }, "part_0.mp4", obj_job.id)
    ]
    
    asr_tags_bucket1 = [
        Tag(3001, 6000, "This is a test.", {}, "part_0.mp4", asr_job.id),
        Tag(1, 3000, "Hello world", {}, "part_0.mp4", asr_job.id),
        Tag(6001, 9000, "How are you?", {}, "part_0.mp4", asr_job.id)
    ]
    
    shot_tags_bucket1 = [
        Tag(0, 100000, "", {}, "part_0.mp4", shot_job.id),
        Tag(100000, 200000, "", {}, "part_0.mp4", shot_job.id)
    ]
    
    # Second bucket: 300,000-600,000ms (5-10 minutes)
    obj_tags_bucket2 = [
        Tag(350000, 355000, "bicycle", {
            "frame_tags": {
                "352000": {"box": {"x1": 20, "y1": 30, "x2": 40, "y2": 50}, "confidence": 0.85}
            }
        }, "part_1.mp4", obj_job.id)
    ]
    
    asr_tags_bucket2 = [
        Tag(320000, 323000, "Second bucket speech", {}, "part_1.mp4", asr_job.id),
        Tag(323000, 326000, "More text here.", {}, "part_1.mp4", asr_job.id)
    ]
    
    tagstore.upload_tags(obj_tags_bucket1 + obj_tags_bucket2, obj_job.id)

    tagstore.upload_tags(asr_tags_bucket1 + asr_tags_bucket2, asr_job.id)

    tagstore.upload_tags(shot_tags_bucket1, shot_job.id)
    
    # Run the upload workflow
    upload_tags_to_fabric(
        source_q=q,
        qwt=q,
        tagstore=tagstore,
        tag_converter=tag_converter
    )
    
    video_tags_metadata = q.content_object_metadata(
        metadata_subtree="video_tags"
    )
    assert isinstance(video_tags_metadata, dict)

    assert "metadata_tags" in video_tags_metadata, "metadata_tags should be present"
    assert "overlay_tags" in video_tags_metadata, "overlay_tags should be present"
    
    assert video_tags_metadata["metadata_tags"]["0000"]["/"] == "./files/video_tags/video-tags-tracks-0000.json"
    assert video_tags_metadata["metadata_tags"]["0001"]["/"] == "./files/video_tags/video-tags-tracks-0001.json"
    assert video_tags_metadata["overlay_tags"]["0000"]["/"] == "./files/video_tags/video-tags-overlay-0000.json"
    assert video_tags_metadata["overlay_tags"]["0001"]["/"] == "./files/video_tags/video-tags-overlay-0001.json"

    # Download and verify the uploaded files
    temp_download_dir = os.path.join(temp_dir, "downloaded_files")
    os.makedirs(temp_download_dir, exist_ok=True)
    
    for bucket_idx, bucket_link in video_tags_metadata["metadata_tags"].items():
        file_path = bucket_link["/"].replace("./files/", "")
        local_path = os.path.join(temp_download_dir, f"metadata_{bucket_idx}.json")

        q.download_file(
            file_path=file_path,
            dest_path=local_path
        )
        
        # Verify file content
        with open(local_path, 'r') as f:
            metadata_content = json.load(f)
        
        assert metadata_content["version"] == 1, "Version should be 1"
        assert "metadata_tags" in metadata_content, "Should have metadata_tags"
        
        metadata_tracks = metadata_content["metadata_tags"]
        
        # Check expected tracks based on bucket
        if bucket_idx == "0000":  # First bucket
            assert "object_detection" in metadata_tracks, "First bucket should have object_detection"
            assert "speech_to_text" in metadata_tracks, "First bucket should have ASR"
            assert "shot_tags" in metadata_tracks, "First bucket should have shot aggregation"
            
            # Verify object detection tags
            obj_tags = metadata_tracks["object_detection"]["tags"]
            assert len(obj_tags) == 2, "First bucket should have 2 object detection tags"
            assert obj_tags[0]["text"] == "person"
            assert obj_tags[1]["text"] == "car"
            
            # Verify ASR tags (converted to auto_captions)
            asr_tags = metadata_tracks["auto_captions"]["tags"]
            assert len(asr_tags) == 2, "Should have auto caption tags"
        
        elif bucket_idx == "0001":  # Second bucket
            assert "object_detection" in metadata_tracks, "Second bucket should have object_detection"
            assert "speech_to_text" in metadata_tracks, "Second bucket should have ASR"
            
            # Verify second bucket content
            obj_tags = metadata_tracks["object_detection"]["tags"]
            assert len(obj_tags) == 1, "Second bucket should have 1 object detection tag"
            assert obj_tags[0]["text"] == "bicycle"
    
    # Download and verify overlay files
    for bucket_idx, bucket_link in video_tags_metadata["overlay_tags"].items():
        file_path = bucket_link["/"].replace("./files/", "")
        local_path = os.path.join(temp_download_dir, f"overlay_{bucket_idx}.json")
        
        q.download_file(
            file_path=file_path,
            dest_path=local_path
        )
        
        # Verify overlay content
        with open(local_path, 'r') as f:
            overlay_content = json.load(f)
        
        assert overlay_content["version"] == 1, "Overlay version should be 1"
        assert "overlay_tags" in overlay_content, "Should have overlay_tags"
        
        frame_tags = overlay_content["overlay_tags"]["frame_level_tags"]
        
        if bucket_idx == "0000":  # First bucket
            # Should have frame tags at timestamps 1000, 3000, 12000
            expected_timestamps = ['1000', '3000', '12000']
            for timestamp in expected_timestamps:
                assert timestamp in frame_tags, f"Frame {timestamp} should be present in first bucket"
                assert "object_detection" in frame_tags[timestamp], f"Frame {timestamp} should have Object Detection"
        
        elif bucket_idx == "0001":  # Second bucket  
            # Should have frame tag at timestamp 352000
            assert '352000' in frame_tags, "Frame 352000 should be present in second bucket"
            assert "object_detection" in frame_tags['352000'], "Frame 352000 should have Object Detection"

def test_upload_tags_empty_tagstore(
    tagstore: Tagstore,
    tag_converter: TagConverter,
    q: Content,
    temp_dir: str
):
    """Test upload workflow with empty tagstore"""
    
    tags_path = os.path.join(temp_dir, "empty_tags")
    os.makedirs(tags_path, exist_ok=True)
    
    # Should handle empty tagstore gracefully
    upload_tags_to_fabric(
        source_q=q,
        qwt=q,
        tagstore=tagstore,
        tag_converter=tag_converter,
    )
    
    # Should not create any files
    assert len(os.listdir(tags_path)) == 0, "No files should be created for empty tagstore"

def test_upload_tags_no_frame_tags(
    tagstore: Tagstore,
    tag_converter: TagConverter,
    q: Content,
    temp_dir: str
):
    """Test upload with only metadata tags (no frame-level tags)"""
    
    qhit = q.qhit
    
    # Create job with only ASR tags (no frame_tags in additional_info)
    asr_job = tagstore.start_job(qhit, "audio", "tagger", "asr")
    
    asr_tags = [
        Tag(0, 3000, "Speech only test", {}, "part_0.mp4", asr_job.id),
        Tag(3000, 6000, "No frame tags here.", {}, "part_0.mp4", asr_job.id)
    ]
    
    for tag in asr_tags:
        tagstore.upload_tags([tag], asr_job.id)
    
    tags_path = os.path.join(temp_dir, "metadata_only")
    os.makedirs(tags_path, exist_ok=True)
    
    upload_tags_to_fabric(
        source_q=q,
        qwt=q,
        tagstore=tagstore,
        tag_converter=tag_converter,
    )
    
    video_tags_metadata = q.content_object_metadata(
        metadata_subtree="video_tags"
    )

    assert isinstance(video_tags_metadata, dict)
    
    assert "metadata_tags" in video_tags_metadata, "Should have metadata_tags"
    # overlay_tags might not be present if no frame-level tags exist
    if "overlay_tags" in video_tags_metadata:
        assert len(video_tags_metadata["overlay_tags"]) == 0, "Should have no overlay files"
    