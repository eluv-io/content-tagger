import pytest
import tempfile
import shutil
import os
import json
import time
from src.tags.tagstore.tagstore import FilesystemTagStore, Tag, UploadJob


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def tag_store(temp_dir):
    """Create a FilesystemTagStore instance for testing"""
    return FilesystemTagStore(temp_dir)


@pytest.fixture
def job_args():
    """Create a sample job for testing"""
    return {
        "qhit": "content-456",
        "stream": "video",
        "track": "llava",
        "author": "test-user"
    }


@pytest.fixture
def sample_tags():
    """Create sample tags for testing"""
    return [
        Tag(100, 200, "person", {"confidence": 0.9}, "llava", "test-job-123"),
        Tag(300, 400, "car", {"confidence": 0.8}, "llava", "test-job-123"),
        Tag(500, 600, "hello world", {"language": "en"}, "asr", "test-job-123"),
        Tag(700, 800, "building", {"type": "office"}, "caption", "test-job-123"),
    ]


def test_init_creates_base_directory(temp_dir):
    """Test that initialization creates the base directory"""
    base_path = os.path.join(temp_dir, "new_dir")
    assert not os.path.exists(base_path)
    
    store = FilesystemTagStore(base_path)
    assert os.path.exists(base_path)
    assert os.path.isdir(base_path)


def test_get_job_dir(tag_store):
    """Test job directory path generation"""
    job_id = "test-job-123"
    expected_path = os.path.join(tag_store.base_path, job_id)
    assert tag_store._get_job_dir(job_id) == expected_path


def test_get_job_metadata_path(tag_store):
    """Test job metadata path generation"""
    job_id = "test-job-123"
    expected_path = os.path.join(tag_store.base_path, job_id, "jobmetadata.json")
    assert tag_store._get_job_metadata_path(job_id) == expected_path


def test_get_tags_path(tag_store):
    """Test tags file path generation"""
    job_id = "test-job-123"
    source = "llava"
    expected_path = os.path.join(tag_store.base_path, job_id, "llava.json")
    assert tag_store._get_tags_path(job_id, source) == expected_path


def test_start_job_creates_directory_and_metadata(tag_store, job_args):
    """Test that starting a job creates directory and metadata file"""
    sample_job = tag_store.start_job(**job_args)

    # Check directory was created
    job_dir = tag_store._get_job_dir(sample_job.id)
    assert os.path.exists(job_dir)
    assert os.path.isdir(job_dir)
    
    # Check metadata file was created
    metadata_path = tag_store._get_job_metadata_path(sample_job.id)
    assert os.path.exists(metadata_path)
    
    # Check metadata content
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    assert metadata['id'] == sample_job.id
    assert metadata['qhit'] == sample_job.qhit
    assert metadata['stream'] == sample_job.stream
    assert metadata['track'] == sample_job.track
    assert metadata['author'] == sample_job.author


def test_get_job_returns_correct_job(tag_store, job_args):
    """Test retrieving job metadata"""
    sample_job = tag_store.start_job(**job_args)

    retrieved_job = tag_store.get_job(sample_job.id)
    
    assert retrieved_job is not None
    assert retrieved_job.id == sample_job.id
    assert retrieved_job.qhit == sample_job.qhit
    assert retrieved_job.stream == sample_job.stream
    assert retrieved_job.track == sample_job.track
    assert retrieved_job.author == sample_job.author


def test_get_job_nonexistent_returns_none(tag_store):
    """Test that getting non-existent job returns None"""
    result = tag_store.get_job("nonexistent-job")
    assert result is None


def test_upload_tags_creates_source_files(tag_store, job_args, sample_tags):
    """Test that uploading tags creates separate files for each source"""
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    # Check that source files were created
    llava_path = tag_store._get_tags_path(sample_job.id, "llava")
    asr_path = tag_store._get_tags_path(sample_job.id, "asr")
    caption_path = tag_store._get_tags_path(sample_job.id, "caption")
    
    assert os.path.exists(llava_path)
    assert os.path.exists(asr_path)
    assert os.path.exists(caption_path)


def test_upload_tags_without_job_raises_error(tag_store, sample_tags):
    """Test that uploading tags without starting job raises error"""
    try:
        tag_store.upload_tags(sample_tags, "test-job-123")
        assert False
    except Exception:
        pass


def test_upload_tags_appends_to_existing(tag_store, job_args):
    """Test that uploading tags appends to existing files"""
    sample_job = tag_store.start_job(**job_args)
    
    # Upload initial tags
    initial_tags = [
        Tag(100, 200, "person", {}, "llava", sample_job.id),
    ]
    tag_store.upload_tags(initial_tags, sample_job.id)
    
    # Verify initial upload worked
    llava_tags = tag_store.get_tags(sample_job.id)
    assert len(llava_tags) == 1
    assert llava_tags[0].text == "person"
    
    # Upload additional tags
    additional_tags = [
        Tag(300, 400, "car", {}, "llava", sample_job.id),
    ]
    tag_store.upload_tags(additional_tags, sample_job.id)
    
    # Check that both tags are present
    llava_tags = tag_store.get_tags(sample_job.id)
    assert len(llava_tags) == 2
    assert {tag.text for tag in llava_tags} == {"person", "car"}


def test_upload_empty_tags_list(tag_store, job_args):
    """Test that uploading empty tags list doesn't create files"""
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags([], sample_job.id)
    
    job_dir = tag_store._get_job_dir(sample_job.id)
    files = os.listdir(job_dir)
    
    # Should only have jobmetadata.json
    assert files == ["jobmetadata.json"]


def test_get_tags_returns_all_tags(tag_store, job_args, sample_tags):
    """Test that get_tags returns tags from all sources"""
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    all_tags = tag_store.get_tags(sample_job.id)
    
    assert len(all_tags) == 4
    sources = {tag.source for tag in all_tags}
    assert sources == {"llava", "asr", "caption"}


def test_get_tags_nonexistent_job_returns_empty(tag_store):
    """Test that getting tags for non-existent job returns empty list"""
    tags = tag_store.get_tags("nonexistent-job")
    assert tags == []


def test_find_jobs_no_filters(tag_store, job_args):
    """Test getting all jobs without filters"""
    sample_job = tag_store.start_job(**job_args)
    
    # Create another job
    job2 = tag_store.start_job(**{"qhit": "content-789", "stream": "audio", "track": "asr", "author": "user2"})
    
    job_ids = tag_store.find_jobs()
    
    assert set(job_ids) == {sample_job.id, job2.id}


def test_find_jobs_with_qhit_filter(tag_store, job_args):
    """Test getting jobs filtered by qhit"""
    sample_job = tag_store.start_job(**job_args)
    
    # Create job with different qhit
    job2 = tag_store.start_job(**{"qhit": "different-content", "stream": "audio", "track": "asr", "author": "user2"})
    job_ids = tag_store.find_jobs(qhit=sample_job.qhit)
    
    assert job_ids == [sample_job.id]


def test_find_jobs_with_track_filter(tag_store, job_args):
    """Test getting jobs filtered by track"""
    sample_job = tag_store.start_job(**job_args)
    
    # Create job with different track
    tag_store.start_job(**{"qhit": "content-789", "stream": "audio", "track": "asr", "author": "user2"})
    
    job_ids = tag_store.find_jobs(track="llava")
    
    assert job_ids == [sample_job.id]


def test_find_jobs_with_multiple_filters(tag_store, job_args):
    """Test getting jobs with multiple filters"""
    sample_job = tag_store.start_job(**job_args)
    
    # Create jobs that match some but not all filters
    job2 = tag_store.start_job(**{"qhit": "content-789", "stream": "audio", "track": "asr", "author": "user2"})
    job3 = tag_store.start_job(**{"qhit": "different-content", "stream": sample_job.stream, "track": sample_job.track, "author": sample_job.author})

    job_ids = tag_store.find_jobs(qhit=sample_job.qhit, track=sample_job.track, author=sample_job.author)

    assert job_ids == [sample_job.id]

def test_upload_tags_handles_missing_job_directory(tag_store, sample_tags):
    """Test proper error when job directory doesn't exist"""
    # Try to upload tags without starting job (no directory exists)
    try:
        tag_store.upload_tags(sample_tags, "test-job-123")
        assert False
    except Exception:
        pass


def test_start_job_and_upload_tags(tag_store, job_args, sample_tags):
    """Test basic job creation and tag upload functionality"""
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    # Check that source files were created
    llava_path = tag_store._get_tags_path(sample_job.id, "llava")
    asr_path = tag_store._get_tags_path(sample_job.id, "asr")
    caption_path = tag_store._get_tags_path(sample_job.id, "caption")
    
    assert os.path.exists(llava_path)
    assert os.path.exists(asr_path)
    assert os.path.exists(caption_path)
    
    # Verify all tags are retrievable
    all_tags = tag_store.get_tags(sample_job.id)
    assert len(all_tags) == 4


def test_find_tags_basic_filters(tag_store, job_args, sample_tags):
    """Test basic tag filtering functionality"""
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    # Test filtering by qhit
    tags = tag_store.find_tags(qhit=sample_job.qhit)
    assert len(tags) == 4
    
    # Test filtering by source
    tags = tag_store.find_tags(sources=["llava"])
    assert len(tags) == 2
    assert all(tag.source == "llava" for tag in tags)
    
    # Test filtering by text content
    tags = tag_store.find_tags(text_contains="hello")
    assert len(tags) == 1
    assert tags[0].text == "hello world"


def test_find_tags_time_range_filters(tag_store, job_args, sample_tags):
    """Test time range filtering"""
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    # Test start_time filters
    tags = tag_store.find_tags(start_time_gte=300)
    assert len(tags) == 3  # Should exclude the first tag (start_time=100)
    
    tags = tag_store.find_tags(start_time_lte=500)
    assert len(tags) == 3  # Should exclude the last tag (start_time=700)


def test_find_jobs_with_filters(tag_store):
    """Test job filtering functionality"""
    # Create multiple jobs
    job1 = UploadJob("job-1", "content-a", "video", "llava", 1000.0, "user1")
    job2 = UploadJob("job-2", "content-b", "audio", "asr", 2000.0, "user2")
    job3 = UploadJob("job-3", "content-a", "video", "caption", 3000.0, "user1")

    job1 = tag_store.start_job(**{"qhit": "content-a", "stream": "video", "track": "llava", "author": "user1"})
    job2 = tag_store.start_job(**{"qhit": "content-b", "stream": "audio", "track": "asr", "author": "user2"})
    job3 = tag_store.start_job(**{"qhit": "content-a", "stream": "video", "track": "caption", "author": "user1"})

    # Test filtering by qhit
    job_ids = tag_store.find_jobs(qhit="content-a")
    assert set(job_ids) == {job1.id, job3.id}

    # Test filtering by author
    job_ids = tag_store.find_jobs(author="user1")
    assert set(job_ids) == {job1.id, job3.id}

    # Test filtering by stream
    job_ids = tag_store.find_jobs(stream="video")
    assert set(job_ids) == {job1.id, job3.id}

    # Test multiple filters
    job_ids = tag_store.find_jobs(qhit="content-a", author="user1")
    assert set(job_ids) == {job1.id, job3.id}


def test_pagination(tag_store, job_args, sample_tags):
    """Test pagination functionality"""
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    # Test limit
    tags = tag_store.find_tags(limit=2)
    assert len(tags) == 2
    
    # Test offset
    tags = tag_store.find_tags(offset=2)
    assert len(tags) == 2
    
    # Test limit + offset
    tags = tag_store.find_tags(limit=1, offset=1)
    assert len(tags) == 1


def test_count_methods(tag_store, job_args, sample_tags):
    """Test counting without loading full data"""
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    # Count all tags
    count = tag_store.count_tags()
    assert count == 4
    
    # Count with filters
    count = tag_store.count_tags(sources=["llava"])
    assert count == 2
    
    # Count jobs
    count = tag_store.count_jobs()
    assert count == 1


def test_error_handling(tag_store, sample_tags):
    """Test error handling for edge cases"""
    # Test uploading tags without starting job
    with pytest.raises(ValueError):
        tag_store.upload_tags(sample_tags, "nonexistent-job")
    
    # Test empty tags upload
    job = tag_store.start_job(**{"qhit": "content", "stream": "video", "track": "track", "author": "user"})
    tag_store.upload_tags([], job.id)  # Should not raise error
    
    # Test getting nonexistent job
    result = tag_store.get_job("nonexistent")
    assert result is None
    
    # Test getting tags for nonexistent job
    tags = tag_store.get_tags("nonexistent")
    assert tags == []


def test_tag_appending(tag_store, job_args):
    """Test that tags are properly appended to existing files"""
    sample_job = tag_store.start_job(**job_args)
    
    # Upload initial tags
    initial_tags = [Tag(100, 200, "person", {}, "llava", sample_job.id)]
    tag_store.upload_tags(initial_tags, sample_job.id)
    
    # Upload additional tags
    additional_tags = [Tag(300, 400, "car", {}, "llava", sample_job.id)]
    tag_store.upload_tags(additional_tags, sample_job.id)
    
    # Check that both tags are present
    all_tags = tag_store.find_tags(job_id=sample_job.id, sources=["llava"])
    assert len(all_tags) == 2
    assert {tag.text for tag in all_tags} == {"person", "car"}


def test_source_with_slash_encoding(tag_store, job_args):
    """Test that sources with slashes are properly base64 encoded"""
    sample_job = tag_store.start_job(**job_args)
    
    # Create tags with sources containing slashes
    tags_with_slashes = [
        Tag(100, 200, "person", {}, "video/segment_1", sample_job.id),
        Tag(300, 400, "car", {}, "audio/track_2", sample_job.id),
        Tag(500, 600, "building", {}, "normal_source", sample_job.id),  # No slash
    ]
    
    tag_store.upload_tags(tags_with_slashes, sample_job.id)
    
    # Verify we can retrieve all tags correctly
    all_tags = tag_store.get_tags(sample_job.id)
    assert len(all_tags) == 3
    
    # Verify original source names are preserved in the tag data
    sources = {tag.source for tag in all_tags}
    assert sources == {"video/segment_1", "audio/track_2", "normal_source"}
    
    # Test filtering by source names with slashes works
    video_tags = tag_store.find_tags(sources=["video/segment_1"])
    assert len(video_tags) == 1
    assert video_tags[0].text == "person"