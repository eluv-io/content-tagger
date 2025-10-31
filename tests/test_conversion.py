import tempfile

import pytest
from src.tags.tagstore.model import UploadJob, Tag
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.tags.conversion import TagConverterConfig, get_latest_tags_for_content
from src.tags.legacy_format import *
from src.tags.conversion import TagConverter, JobWithTags
from unittest.mock import MagicMock, patch

@pytest.fixture
def tag_converter():
    """Create a TagConverter with test configuration"""
    config = TagConverterConfig(
        interval=5,  # 5 minutes
        name_mapping={
            "object_detection": "Object Detection",
            "asr": "Speech to Text",
            "shot": "Shot Detection"
        },
        single_tag_tracks=[],
        coalesce_tracks=["asr"],
        max_sentence_words=200
    )
    return TagConverter(config)

@pytest.fixture
def sample_job_tags():
    """Create sample JobWithTags for testing"""
    # Object detection job
    obj_job = UploadJob(
        id="job1",
        qhit="iq__test",
        stream="video", 
        track="object_detection",
        timestamp=1640995200.0,
        author="tagger"
    )
    
    obj_tags = [
        Tag(0, 5000, "person", {}, "part_0.mp4", "job1"),
        Tag(10000, 15000, "car", {}, "part_0.mp4", "job1"),
        Tag(20000, 25000, "person", {}, "part_0.mp4", "job1"),
        Tag(26000, 29000, "dog", {}, "part_0.mp4", "job1"),
        Tag(30000, 35000, "bicycle", {}, "part_1.mp4", "job1")
    ]
    
    # ASR job
    asr_job = UploadJob(
        id="job2",
        qhit="iq__test",
        stream="audio",
        track="asr", 
        timestamp=1640995300.0,
        author="tagger"
    )
    
    asr_tags = [
        Tag(1, 3000, "Hello world,", {}, "part_0.mp4", "job2"),
        Tag(3001, 6000, "How are you?", {}, "part_0.mp4", "job2"),
        Tag(6001, 9000, "This is a test.", {}, "part_0.mp4", "job2")
    ]
    
    # Shot detection job
    shot_job = UploadJob(
        id="job3",
        qhit="iq__test",
        stream="video",
        track="shot",
        timestamp=1640995400.0,
        author="tagger"
    )
    
    shot_tags = [
        Tag(0, 10000, "", {}, "part_0.mp4", "job3"),   # Shot 1
        Tag(10000, 20000, "", {}, "part_0.mp4", "job3"), # Shot 2
        Tag(20000, 30000, "", {}, "part_0.mp4", "job3"),  # Shot 3
        Tag(30000, 40000, "", {}, "part_1.mp4", "job3")  # Continutation of Shot 3
    ]
    
    return [
        JobWithTags(job=obj_job, tags=obj_tags),
        JobWithTags(job=asr_job, tags=asr_tags),
        JobWithTags(job=shot_job, tags=shot_tags)
    ]

def test_get_latest_tags_complex_deduplication():
    """Test get_latest_tags_for_content with multiple tracks per source and job shadowing"""
    
    # Create jobs with overlapping source+track combinations
    # Newer jobs should shadow older ones for the same source+track
    jobs = [
        # Initial jobs for content
        UploadJob("job1_old", "iq__test", "video", "object_detection", 1640995100.0, "tagger"),
        UploadJob("job2_old", "iq__test", "audio", "asr", 1640995150.0, "tagger"), 
        
        # Newer job that should shadow the object_detection from job1_old
        UploadJob("job3_new", "iq__test", "video", "object_detection", 1640995200.0, "tagger"),
        
        # Job with different track for same source as job1
        UploadJob("job4_face", "iq__test", "video", "face_detection", 1640995180.0, "tagger"),
        
        # Another ASR job that should shadow job2_old
        UploadJob("job5_asr_new", "iq__test", "audio", "asr", 1640995250.0, "tagger"),
    ]
    
    # Create tags for different sources and jobs
    tags = [
        Tag(0, 1000, "old_person", {"frame_tags": {"500": {"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.8}}}, 
            "part_0.mp4", "job1_old"),
        Tag(2000, 3000, "old_person_2", {"frame_tags": {"2500": {"box": {"x1": 12, "y1": 22, "x2": 32, "y2": 42}, "confidence": 0.7}}}, 
            "part_0.mp4", "job1_old"),  # SECOND tag on same source from same job
        Tag(5000, 6000, "old_car", {}, "part_1.mp4", "job1_old"),
        
        Tag(0, 1000, "old speech one", {}, "part_0.mp4", "job2_old"),
        Tag(1000, 2000, "old speech two", {}, "part_0.mp4", "job2_old"),  # SECOND ASR tag on same source
        
        Tag(0, 1000, "new_person", {"frame_tags": {"500": {"box": {"x1": 15, "y1": 25, "x2": 35, "y2": 45}, "confidence": 0.9}}}, 
            "part_0.mp4", "job3_new"),
        Tag(1500, 2500, "new_person_2", {"frame_tags": {"2000": {"box": {"x1": 17, "y1": 27, "x2": 37, "y2": 47}, "confidence": 0.85}}}, 
            "part_0.mp4", "job3_new"),  # SECOND tag on same source from newer job
        
        Tag(2000, 3000, "face_detected", {"frame_tags": {"2500": {"box": {"x1": 50, "y1": 60, "x2": 70, "y2": 80}, "confidence": 0.95}}}, 
            "part_0.mp4", "job4_face"),
        Tag(3500, 4500, "face_detected_2", {"frame_tags": {"4000": {"box": {"x1": 52, "y1": 62, "x2": 72, "y2": 82}, "confidence": 0.92}}}, 
            "part_0.mp4", "job4_face"),  # SECOND face tag on same source
        
        Tag(0, 1000, "new speech one", {}, "part_0.mp4", "job5_asr_new"),
        Tag(1000, 2000, "new speech two", {}, "part_0.mp4", "job5_asr_new"),  # SECOND tag on part_0.mp4
        Tag(10000, 11000, "more speech one", {}, "part_1.mp4", "job5_asr_new"),
        Tag(11000, 12000, "more speech two", {}, "part_1.mp4", "job5_asr_new"),  # SECOND tag on part_1.mp4
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tagstore = FilesystemTagStore(temp_dir)
        
        with patch.object(tagstore, 'find_jobs') as mock_find, \
             patch.object(tagstore, 'get_job') as mock_get_job, \
             patch.object(tagstore, 'find_tags') as mock_get_tags:
            
            mock_find.return_value = ["job1_old", "job2_old", "job3_new", "job4_face", "job5_asr_new"]
            mock_get_job.side_effect = lambda jobid, q=None: next((job for job in jobs if job.id == jobid), None)
            mock_get_tags.side_effect = lambda q=None, **kwargs: [tag for tag in tags if tag.jobid == kwargs.get('jobid', tag.jobid)]

            #result = get_latest_tags_for_content(MagicMock(qhit="iq__test"), tagstore)
            result = get_latest_tags_for_content(MagicMock(qhit="iq__test"), tagstore)
            
            # Should have 5 jobs returned (all jobs, but with filtered tags)
            assert len(result) == 5
            
            # Sort by job timestamp to make assertions easier
            result_by_job = {item.job.id: item for item in result}
            
            # job5_asr_new (newest ASR) - should have ALL 4 tags (2 on each source)
            job5_tags = result_by_job["job5_asr_new"]
            assert len(job5_tags.tags) == 4  # Changed from 2 to 4
            expected_texts = {"new speech one", "new speech two", "more speech one", "more speech two"}
            assert {tag.text for tag in job5_tags.tags} == expected_texts
            assert {tag.source for tag in job5_tags.tags} == {"part_0.mp4", "part_1.mp4"}
            
            # Check that we have 2 tags per source
            part0_tags = [tag for tag in job5_tags.tags if tag.source == "part_0.mp4"]
            part1_tags = [tag for tag in job5_tags.tags if tag.source == "part_1.mp4"]
            assert len(part0_tags) == 2, "Should have 2 tags on part_0.mp4"
            assert len(part1_tags) == 2, "Should have 2 tags on part_1.mp4"
            
            # job3_new (newer object_detection) - should have BOTH tags for part_0.mp4
            job3_tags = result_by_job["job3_new"]
            assert len(job3_tags.tags) == 2  # Changed from 1 to 2
            expected_obj_texts = {"new_person", "new_person_2"}
            assert {tag.text for tag in job3_tags.tags} == expected_obj_texts
            assert all(tag.source == "part_0.mp4" for tag in job3_tags.tags)
            
            # job4_face (face_detection) - should have BOTH face tags (no conflicts)
            job4_tags = result_by_job["job4_face"] 
            assert len(job4_tags.tags) == 2  # Changed from 1 to 2
            expected_face_texts = {"face_detected", "face_detected_2"}
            assert {tag.text for tag in job4_tags.tags} == expected_face_texts
            assert all(tag.source == "part_0.mp4" for tag in job4_tags.tags)
            
            # job1_old (old object_detection) - should only have part_1.mp4 tag 
            # (part_0.mp4 tags shadowed by job3, but part_1.mp4 tag preserved)
            job1_tags = result_by_job["job1_old"]
            assert len(job1_tags.tags) == 1  # Only part_1.mp4 tag remains
            assert job1_tags.tags[0].text == "old_car"
            assert job1_tags.tags[0].source == "part_1.mp4"
            
            job2_tags = result_by_job["job2_old"]
            assert len(job2_tags.tags) == 0
            
            total_tags = sum(len(item.tags) for item in result)
            assert total_tags == 9, f"Expected 9 total tags, got {total_tags}"  # 4 + 2 + 2 + 1 + 0
            
            # Verify the source+track combinations that should be present
            expected_combinations = {
                ("part_0.mp4", "object_detection"),  # From job3_new (shadows job1_old) - 2 tags
                ("part_1.mp4", "object_detection"),  # From job1_old (no shadowing) - 1 tag
                ("part_0.mp4", "face_detection"),    # From job4_face (unique track) - 2 tags
                ("part_0.mp4", "asr"),               # From job5_asr_new (shadows job2_old) - 2 tags
                ("part_1.mp4", "asr"),               # From job5_asr_new (unique) - 2 tags
            }
            
            actual_combinations = set()
            for item in result:
                for tag in item.tags:
                    actual_combinations.add((tag.source, item.job.track))
            
            assert actual_combinations == expected_combinations

def test_get_tracks_basic_conversion(tag_converter, sample_job_tags):
    """Test basic conversion from JobWithTags to TrackCollection"""
    track_collection = tag_converter.get_tracks(sample_job_tags)
    
    # Check that we have a TrackCollection
    assert isinstance(track_collection, TrackCollection)
    assert hasattr(track_collection, 'tracks')
    assert hasattr(track_collection, 'agg_tracks')
    
    # Check individual tracks were created
    assert "object_detection" in track_collection.tracks
    assert "asr" in track_collection.tracks
    assert "shot" in track_collection.tracks
    
    # Check object detection track
    obj_track = track_collection.tracks["object_detection"]
    assert len(obj_track) == 5
    assert obj_track[0].start_time == 0
    assert obj_track[0].end_time == 5000
    assert obj_track[0].text == "person"
    
    # Check ASR track
    asr_track = track_collection.tracks["asr"]
    assert len(asr_track) == 3
    assert asr_track[0].text == "Hello world,"
    assert asr_track[1].text == "How are you?"
    assert asr_track[2].text == "This is a test."

def test_get_tracks_shot_aggregation(tag_converter, sample_job_tags):
    """Test that shot intervals create aggregated tags"""
    track_collection = tag_converter.get_tracks(sample_job_tags)
    
    # Should have shot_tags aggregation since we have shot
    assert "shot_tags" in track_collection.agg_tracks
    
    shot_agg_tags = track_collection.agg_tracks["shot_tags"]
    assert len(shot_agg_tags) == 3  # 3 shots
    
    # First shot should contain object detection tags from 0-10000ms
    first_shot = shot_agg_tags[0]
    assert first_shot.start_time == 0
    assert first_shot.end_time == 10000
    assert "object_detection" in first_shot.tags
    assert len(first_shot.tags["object_detection"]) == 1  # 1 person tag in first shot
    assert first_shot.tags["object_detection"][0].text == "person"

    # Last shot should contain two object tags and begin at 20s and end at 40s
    last_shot = shot_agg_tags[-1]
    assert last_shot.start_time == 20000
    assert last_shot.end_time == 40000
    assert "object_detection" in last_shot.tags
    assert len(last_shot.tags["object_detection"]) == 3

def test_get_tracks_asr_auto_captions(tag_converter, sample_job_tags):
    """Test that ASR creates auto_captions track"""
    track_collection = tag_converter.get_tracks(sample_job_tags)
    
    assert "auto_captions" in track_collection.tracks
    
    auto_captions = track_collection.tracks["auto_captions"]
    
    assert len(auto_captions) == 2
    
    assert auto_captions[0].text == "Hello world, How are you?"
    assert auto_captions[0].start_time == 1
    assert auto_captions[0].end_time == 6001
    assert auto_captions[1].text == "This is a test."
    assert auto_captions[1].start_time == 6001

def test_dump_tracks_basic_structure(tag_converter, sample_job_tags):
    """Test dump_tracks creates correct JSON structure"""
    track_collection = tag_converter.get_tracks(sample_job_tags)
    result = tag_converter.dump_tracks(track_collection)
    
    # Check top-level structure
    assert "version" in result
    assert result["version"] == 1
    assert "metadata_tags" in result
    
    metadata_tags = result["metadata_tags"]
    
    # Check that tracks were converted with proper labels
    assert "object_detection" in metadata_tags
    assert "speech_to_text" in metadata_tags
    assert "shot_tags" in metadata_tags
    assert "auto_captions" in metadata_tags
    assert "shot_detection" in metadata_tags

    # Check object detection structure
    obj_detection = metadata_tags["object_detection"]
    assert obj_detection["label"] == "Object Detection"
    assert "tags" in obj_detection
    assert len(obj_detection["tags"]) == 5

def test_dump_tracks_aggregated_tags(tag_converter, sample_job_tags):
    """Test that aggregated tags are formatted correctly"""
    track_collection = tag_converter.get_tracks(sample_job_tags)
    result = tag_converter.dump_tracks(track_collection)
    
    # Should have shot_tags in metadata_tags
    assert "shot_tags" in result["metadata_tags"]
    
    shot_tags = result["metadata_tags"]["shot_tags"]["tags"]
    assert len(shot_tags) == 3  # 3 shots
    
    # Check aggregated tag structure
    first_shot_tag = shot_tags[0]
    assert "start_time" in first_shot_tag
    assert "end_time" in first_shot_tag
    assert "text" in first_shot_tag
    assert first_shot_tag["start_time"] == 0
    assert first_shot_tag["end_time"] == 10000
    
    # Check that text contains aggregated features
    assert isinstance(first_shot_tag["text"], dict)
    assert "Object Detection" in first_shot_tag["text"]  # Feature label
    
    # Check that aggregated tags have proper format
    obj_detection_in_shot = first_shot_tag["text"]["Object Detection"]
    assert isinstance(obj_detection_in_shot, list)
    assert len(obj_detection_in_shot) == 1
    
    # Tags should be wrapped in arrays per convention
    first_obj_tag = obj_detection_in_shot[0]
    assert "text" in first_obj_tag
    assert isinstance(first_obj_tag["text"], list)  # Text wrapped in array
    assert first_obj_tag["text"] == ["person"]

def test_dump_tracks_empty_input(tag_converter):
    """Test dump_tracks with empty TrackCollection"""
    empty_collection = TrackCollection(tracks={}, agg_tracks={})
    result = tag_converter.dump_tracks(empty_collection)
    
    assert result["version"] == 1
    assert result["metadata_tags"] == {}

def test_get_overlays_basic_conversion(tag_converter):
    """Test basic conversion from JobWithTags to Overlay format"""
    # Create job with frame-level tags
    obj_job = UploadJob("job1", "iq__test", "video", "object_detection", 1640995200.0, "tagger")
    
    obj_tags = [
        Tag(0, 5000, "person", {
            "frame_tags": {
                "1000": {"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.9},
                "3000": {"box": {"x1": 15, "y1": 25, "x2": 35, "y2": 45}, "confidence": 0.8}
            }
        }, "part_0.mp4", "job1"),
        Tag(5000, 10000, "car", {
            "frame_tags": {
                "7000": {"box": {"x1": 50, "y1": 60, "x2": 70, "y2": 80}, "confidence": 0.95}
            }
        }, "part_0.mp4", "job1")
    ]
    
    job_tags = [JobWithTags(job=obj_job, tags=obj_tags)]
    
    overlay = tag_converter.get_overlays(job_tags)
    
    # Check that we have frame-level data
    assert isinstance(overlay, dict)
    assert 1000 in overlay
    assert 3000 in overlay
    assert 7000 in overlay
    
    # Check frame 1000
    frame_1000 = overlay[1000]
    assert "object_detection" in frame_1000
    assert len(frame_1000["object_detection"]) == 1
    
    frame_tag = frame_1000["object_detection"][0]
    assert frame_tag.text == "person"
    assert frame_tag.box == {"x1": 10, "y1": 20, "x2": 30, "y2": 40}
    assert frame_tag.confidence == 0.9
    
    # Check frame 7000
    frame_7000 = overlay[7000]
    assert "object_detection" in frame_7000
    assert len(frame_7000["object_detection"]) == 1
    
    frame_tag = frame_7000["object_detection"][0]
    assert frame_tag.text == "car"
    assert frame_tag.box == {"x1": 50, "y1": 60, "x2": 70, "y2": 80}
    assert frame_tag.confidence == 0.95

def test_get_overlays_multiple_features(tag_converter):
    """Test overlay conversion with multiple features in same frame"""
    # Create jobs for different features
    obj_job = UploadJob("job1", "iq__test", "video", "object_detection", 1640995200.0, "tagger")
    face_job = UploadJob("job2", "iq__test", "video", "face_detection", 1640995300.0, "tagger")
    
    obj_tags = [
        Tag(0, 5000, "person", {
            "frame_tags": {
                "1000": {"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.9}
            }
        }, "part_0.mp4", "job1")
    ]
    
    face_tags = [
        Tag(0, 5000, "face", {
            "frame_tags": {
                "1000": {"box": {"x1": 12, "y1": 22, "x2": 28, "y2": 38}, "confidence": 0.85}
            }
        }, "part_0.mp4", "job2")
    ]
    
    job_tags = [
        JobWithTags(job=obj_job, tags=obj_tags),
        JobWithTags(job=face_job, tags=face_tags)
    ]
    
    overlay = tag_converter.get_overlays(job_tags)
    
    # Frame 1000 should have both features
    frame_1000 = overlay[1000]
    assert "object_detection" in frame_1000
    assert "face_detection" in frame_1000
    
    # Check object detection
    obj_tag = frame_1000["object_detection"][0]
    assert obj_tag.text == "person"
    assert obj_tag.box == {"x1": 10, "y1": 20, "x2": 30, "y2": 40}

    # Check face detection
    face_tag = frame_1000["face_detection"][0]
    assert face_tag.text == "face"
    assert face_tag.box == {"x1": 12, "y1": 22, "x2": 28, "y2": 38}

def test_get_overlays_empty_input(tag_converter):
    """Test overlay conversion with no frame tags"""
    job_tags = []
    overlay = tag_converter.get_overlays(job_tags)
    assert overlay == {}

def test_get_overlays_no_frame_tags(tag_converter):
    """Test overlay conversion with tags that have no frame_tags"""
    obj_job = UploadJob("job1", "iq__test", "video", "object_detection", 1640995200.0, "tagger")
    
    # Tags without frame_tags in additional_info
    obj_tags = [
        Tag(0, 5000, "person", {}, "part_0.mp4", "job1"),
        Tag(5000, 10000, "car", {"other_data": "value"}, "part_0.mp4", "job1")
    ]
    
    job_tags = [JobWithTags(job=obj_job, tags=obj_tags)]
    
    overlay = tag_converter.get_overlays(job_tags)
    assert overlay == {}

def test_dump_overlay_basic_structure(tag_converter):
    """Test dump_overlay creates correct JSON structure"""
    # Create simple overlay data
    overlay = {
        1000: {
            "object_detection": [
                FrameTag.from_dict({"text": "person", "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.9})
            ]
        },
        3000: {
            "object_detection": [
                FrameTag.from_dict({"text": "car", "box": {"x1": 50, "y1": 60, "x2": 70, "y2": 80}, "confidence": 0.95})
            ],
            "face_detection": [
                FrameTag.from_dict({"text": "face", "box": {"x1": 15, "y1": 25, "x2": 35, "y2": 45}, "confidence": 0.8})
            ]
        }
    }
    
    result = tag_converter.dump_overlay(overlay)
    
    # Check top-level structure
    assert "version" in result
    assert result["version"] == 1
    assert "overlay_tags" in result
    
    overlay_tags = result["overlay_tags"]["frame_level_tags"]
    
    # Check frame 1000
    assert '1000' in overlay_tags
    frame_1000 = overlay_tags['1000']
    assert "object_detection" in frame_1000  # Feature name mapped

    obj_tags = frame_1000["object_detection"]
    assert len(obj_tags) == 1
    assert obj_tags["tags"][0]["text"] == "person"
    assert obj_tags["tags"][0]["box"] == {"x1": 10, "y1": 20, "x2": 30, "y2": 40}
    assert obj_tags["tags"][0]["confidence"] == 0.9
    
    # Check frame 3000 has both features
    assert '3000' in overlay_tags
    frame_3000 = overlay_tags['3000']
    assert "object_detection" in frame_3000
    assert "face_detection" in frame_3000

def test_dump_overlay_empty_input(tag_converter):
    """Test dump_overlay with empty overlay"""
    empty_overlay = {}
    result = tag_converter.dump_overlay(empty_overlay)
    
    assert result["version"] == 1
    assert result["overlay_tags"] == {"frame_level_tags": {}}