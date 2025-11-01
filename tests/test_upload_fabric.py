import pytest
import os
import json
from dotenv import load_dotenv

from src.tags.conversion_workflow import upload_tags_to_fabric
from src.tags.conversion import TagConverter, TagConverterConfig
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Tag
from src.common.content import Content

load_dotenv()

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

def test_upload_tags_to_fabric_full_workflow(
    tag_store: Tagstore,
    tag_converter: TagConverter,
    writable_q: Content,
    temp_dir: str,
):
    """Test complete upload workflow with multiple tracks and time buckets"""
    q = writable_q

    qhit = q.qid

    # Create test jobs in tagstore
    obj_job = tag_store.create_batch(qhit, track="object", author="tagger", stream="video", q=q)
    asr_job = tag_store.create_batch(qhit, stream="audio", author="tagger", track="asr", q=q)
    shot_job = tag_store.create_batch(qhit, stream="video", author="tagger", track="shot", q=q)

    # Create tags spanning multiple time buckets (5-minute intervals = 300,000ms)
    # First bucket: 0-300,000ms
    obj_tags_bucket1 = [
        Tag(0, 5000, "person", {
                "1000": {"box": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}, "confidence": 0.9},
                "3000": {"box": {"x1": 0.15, "y1": 0.25, "x2": 0.35, "y2": 0.45}, "confidence": 0.8}
            }, {}, "part_0.mp4", obj_job.id),
        Tag(10000, 15000, "car", {
                "12000": {"box": {"x1": 0.5, "y1": 0.6, "x2": 0.7, "y2": 0.8}, "confidence": 0.95}
            }, {}, "part_0.mp4", obj_job.id)
    ]
    
    asr_tags_bucket1 = [
        Tag(3001, 6000, "This is a test.", {}, {}, "part_0.mp4", asr_job.id),
        Tag(1, 3000, "Hello world", {}, {}, "part_0.mp4", asr_job.id),
        Tag(6001, 9000, "How are you?", {}, {}, "part_0.mp4", asr_job.id)
    ]
    
    shot_tags_bucket1 = [
        Tag(0, 100000, "", {}, {}, "part_0.mp4", shot_job.id),
        Tag(100000, 200000, "", {}, {}, "part_0.mp4", shot_job.id)
    ]
    
    # Second bucket: 300,000-600,000ms (5-10 minutes)
    obj_tags_bucket2 = [
        Tag(350000, 355000, "bicycle", {
                "352000": {"box": {"x1": 0.2, "y1": 0.3, "x2": 0.4, "y2": 0.5}, "confidence": 0.85}
            }, {}, "part_1.mp4", obj_job.id)
    ]
    
    asr_tags_bucket2 = [
        Tag(320000, 323000, "Second bucket speech", {}, {}, "part_1.mp4", asr_job.id),
        Tag(323000, 326000, "More text here.", {}, {}, "part_1.mp4", asr_job.id)
    ]
    
    tag_store.upload_tags(obj_tags_bucket1 + obj_tags_bucket2, obj_job.id, q=q)
    tag_store.upload_tags(asr_tags_bucket1 + asr_tags_bucket2, asr_job.id, q=q)
    tag_store.upload_tags(shot_tags_bucket1, shot_job.id, q=q)
    
    # Run the upload workflow
    upload_tags_to_fabric(
        source_q=q,
        qwt=q,
        tagstore=tag_store,
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
    tag_store: Tagstore,
    tag_converter: TagConverter,
    writable_q: Content,
    temp_dir: str
):
    """Test upload workflow with empty tagstore"""
    q = writable_q
    
    tags_path = os.path.join(temp_dir, "empty_tags")
    os.makedirs(tags_path, exist_ok=True)
    
    # Should handle empty tagstore gracefully
    upload_tags_to_fabric(
        source_q=q,
        qwt=q,
        tagstore=tag_store,
        tag_converter=tag_converter,
    )
    
    # Should not create any files
    assert len(os.listdir(tags_path)) == 0, "No files should be created for empty tagstore"

def test_upload_tags_no_frame_tags(
    tag_store: Tagstore,
    tag_converter: TagConverter,
    writable_q: Content,
    temp_dir: str
):
    """Test upload with only metadata tags (no frame-level tags)"""
    q = writable_q
    
    # Create job with only ASR tags (no frame_tags in additional_info)
    asr_job = tag_store.create_batch(q.qid, "audio", "tagger", "asr", q=q)
    
    asr_tags = [
        Tag(0, 3000, "Speech only test", {}, {}, "part_0.mp4", asr_job.id),
        Tag(3000, 6000, "No frame tags here.", {}, {}, "part_0.mp4", asr_job.id)
    ]
    
    for tag in asr_tags:
        tag_store.upload_tags([tag], asr_job.id, q=q)
    
    tags_path = os.path.join(temp_dir, "metadata_only")
    os.makedirs(tags_path, exist_ok=True)
    
    upload_tags_to_fabric(
        source_q=q,
        qwt=q,
        tagstore=tag_store,
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
