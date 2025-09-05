import tempfile
from src.tags.tagstore.types import UploadJob, Tag
from src.tags.tagstore.tagstore import FilesystemTagStore
from src.tags.conversion import get_latest_tags_for_content
from src.tags.tagstore.types import TagStoreConfig
from unittest.mock import patch

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
        # job1_old - object_detection for part_0.mp4 and part_1.mp4 (should be shadowed for part_0 only)
        Tag(0, 1000, "old_person", {"frame_tags": {"500": {"box": [10, 20, 30, 40], "confidence": 0.8}}}, 
            "part_0.mp4", "job1_old"),
        Tag(5000, 6000, "old_car", {}, "part_1.mp4", "job1_old"),
        
        # job2_old - ASR for part_0.mp4 (should be completely shadowed)
        Tag(0, 2000, "old speech", {}, "part_0.mp4", "job2_old"),
        
        # job3_new - object_detection for part_0.mp4 only (shadows job1_old for this source)
        Tag(0, 1000, "new_person", {"frame_tags": {"500": {"box": [15, 25, 35, 45], "confidence": 0.9}}}, 
            "part_0.mp4", "job3_new"),
        
        # job4_face - face_detection for part_0.mp4 (new track, no shadowing)
        Tag(2000, 3000, "face_detected", {"frame_tags": {"2500": {"box": [50, 60, 70, 80], "confidence": 0.95}}}, 
            "part_0.mp4", "job4_face"),
        
        # job5_asr_new - ASR for part_0.mp4 and part_1.mp4 (shadows job2_old)
        Tag(0, 2000, "new speech", {}, "part_0.mp4", "job5_asr_new"),
        Tag(10000, 12000, "more speech", {}, "part_1.mp4", "job5_asr_new"),
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = TagStoreConfig(base_dir=temp_dir)
        tagstore = FilesystemTagStore(config)
        
        with patch.object(tagstore, 'find_jobs') as mock_find, \
             patch.object(tagstore, 'get_job') as mock_get_job, \
             patch.object(tagstore, 'get_tags') as mock_get_tags:
            
            mock_find.return_value = ["job1_old", "job2_old", "job3_new", "job4_face", "job5_asr_new"]
            mock_get_job.side_effect = lambda job_id: next((job for job in jobs if job.id == job_id), None)
            mock_get_tags.side_effect = lambda job_id: [tag for tag in tags if tag.jobid == job_id]
            
            result = get_latest_tags_for_content("iq__test", tagstore)
            
            # Should have 5 jobs returned (all jobs, but with filtered tags)
            assert len(result) == 5
            
            # Sort by job timestamp to make assertions easier
            result_by_job = {item.job.id: item for item in result}
            
            # job5_asr_new (newest ASR) - should have both tags
            job5_tags = result_by_job["job5_asr_new"]
            assert len(job5_tags.tags) == 2
            assert {tag.text for tag in job5_tags.tags} == {"new speech", "more speech"}
            assert {tag.source for tag in job5_tags.tags} == {"part_0.mp4", "part_1.mp4"}
            
            # job3_new (newer object_detection) - should have its tag for part_0.mp4
            job3_tags = result_by_job["job3_new"]
            assert len(job3_tags.tags) == 1
            assert job3_tags.tags[0].text == "new_person"
            assert job3_tags.tags[0].source == "part_0.mp4"
            
            # job4_face (face_detection) - should have its tag (no conflicts)
            job4_tags = result_by_job["job4_face"] 
            assert len(job4_tags.tags) == 1
            assert job4_tags.tags[0].text == "face_detected"
            assert job4_tags.tags[0].source == "part_0.mp4"
            
            # job1_old (old object_detection) - should only have part_1.mp4 tag (part_0.mp4 shadowed by job3)
            job1_tags = result_by_job["job1_old"]
            assert len(job1_tags.tags) == 1
            assert job1_tags.tags[0].text == "old_car"
            assert job1_tags.tags[0].source == "part_1.mp4"
            
            # job2_old (old ASR) - should have no tags (completely shadowed by job5)
            job2_tags = result_by_job["job2_old"]
            assert len(job2_tags.tags) == 0
            
            # Verify the source+track combinations that should be present
            expected_combinations = {
                ("part_0.mp4", "object_detection"),  # From job3_new (shadows job1_old)
                ("part_1.mp4", "object_detection"),  # From job1_old (no shadowing)
                ("part_0.mp4", "face_detection"),    # From job4_face (unique track)
                ("part_0.mp4", "asr"),               # From job5_asr_new (shadows job2_old)
                ("part_1.mp4", "asr"),               # From job5_asr_new (unique)
            }
            
            actual_combinations = set()
            for item in result:
                for tag in item.tags:
                    actual_combinations.add((tag.source, item.job.track))
            
            assert actual_combinations == expected_combinations