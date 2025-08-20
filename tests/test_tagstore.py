import pytest
import tempfile
import shutil
import os
import json
import time
from src.tags.tagstore import FilesystemTagStore, Tag, Job


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
def sample_job():
    """Create a sample job for testing"""
    return Job(
        id="test-job-123",
        qhit="content-456",
        stream="video",
        feature="llava",
        timestamp=time.time(),
        author="test-user"
    )


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


def test_start_job_creates_directory_and_metadata(tag_store, sample_job):
    """Test that starting a job creates directory and metadata file"""
    tag_store.start_job(sample_job)
    
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
    assert metadata['feature'] == sample_job.feature
    assert metadata['author'] == sample_job.author


def test_get_job_returns_correct_job(tag_store, sample_job):
    """Test retrieving job metadata"""
    tag_store.start_job(sample_job)
    
    retrieved_job = tag_store.get_job(sample_job.id)
    
    assert retrieved_job is not None
    assert retrieved_job.id == sample_job.id
    assert retrieved_job.qhit == sample_job.qhit
    assert retrieved_job.stream == sample_job.stream
    assert retrieved_job.feature == sample_job.feature
    assert retrieved_job.author == sample_job.author


def test_get_job_nonexistent_returns_none(tag_store):
    """Test that getting non-existent job returns None"""
    result = tag_store.get_job("nonexistent-job")
    assert result is None


def test_upload_tags_creates_source_files(tag_store, sample_job, sample_tags):
    """Test that uploading tags creates separate files for each source"""
    tag_store.start_job(sample_job)
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


def test_upload_tags_appends_to_existing(tag_store, sample_job):
    """Test that uploading tags appends to existing files"""
    tag_store.start_job(sample_job)
    
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


def test_upload_empty_tags_list(tag_store, sample_job):
    """Test that uploading empty tags list doesn't create files"""
    tag_store.start_job(sample_job)
    tag_store.upload_tags([], sample_job.id)
    
    job_dir = tag_store._get_job_dir(sample_job.id)
    files = os.listdir(job_dir)
    
    # Should only have jobmetadata.json
    assert files == ["jobmetadata.json"]


def test_get_tags_returns_all_tags(tag_store, sample_job, sample_tags):
    """Test that get_tags returns tags from all sources"""
    tag_store.start_job(sample_job)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    all_tags = tag_store.get_tags(sample_job.id)
    
    assert len(all_tags) == 4
    sources = {tag.source for tag in all_tags}
    assert sources == {"llava", "asr", "caption"}


def test_get_tags_nonexistent_job_returns_empty(tag_store):
    """Test that getting tags for non-existent job returns empty list"""
    tags = tag_store.get_tags("nonexistent-job")
    assert tags == []


def test_get_jobs_no_filters(tag_store, sample_job):
    """Test getting all jobs without filters"""
    tag_store.start_job(sample_job)
    
    # Create another job
    job2 = Job("job-456", "content-789", "audio", "asr", time.time(), "user2")
    tag_store.start_job(job2)
    
    job_ids = tag_store.get_jobs()
    
    assert set(job_ids) == {sample_job.id, job2.id}


def test_get_jobs_with_qhit_filter(tag_store, sample_job):
    """Test getting jobs filtered by qhit"""
    tag_store.start_job(sample_job)
    
    # Create job with different qhit
    job2 = Job("job-456", "different-content", "audio", "asr", time.time(), "user2")
    tag_store.start_job(job2)
    
    job_ids = tag_store.get_jobs(qhit=sample_job.qhit)
    
    assert job_ids == [sample_job.id]


def test_get_jobs_with_feature_filter(tag_store, sample_job):
    """Test getting jobs filtered by feature"""
    tag_store.start_job(sample_job)
    
    # Create job with different feature
    job2 = Job("job-456", "content-789", "audio", "asr", time.time(), "user2")
    tag_store.start_job(job2)
    
    job_ids = tag_store.get_jobs(feature="llava")
    
    assert job_ids == [sample_job.id]


def test_get_jobs_with_multiple_filters(tag_store, sample_job):
    """Test getting jobs with multiple filters"""
    tag_store.start_job(sample_job)
    
    # Create jobs that match some but not all filters
    job2 = Job("job-456", sample_job.qhit, "audio", "asr", time.time(), "user2")
    job3 = Job("job-789", "different-content", sample_job.stream, sample_job.feature, time.time(), sample_job.author)
    tag_store.start_job(job2)
    tag_store.start_job(job3)
    
    job_ids = tag_store.get_jobs(qhit=sample_job.qhit, feature=sample_job.feature, auth=sample_job.author)
    
    assert job_ids == [sample_job.id]


def test_start_job_creates_directory_with_existing_dir(tag_store):
    """Test that starting a job works even if directory exists"""
    job = Job("test-job", "content", "video", "llava", time.time(), "user")
    
    # Create the directory first
    job_dir = tag_store._get_job_dir(job.id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Should not raise an error
    tag_store.start_job(job)
    
    # Should still create metadata
    metadata_path = tag_store._get_job_metadata_path(job.id)
    assert os.path.exists(metadata_path)


def test_upload_tags_handles_missing_job_directory(tag_store, sample_tags):
    """Test proper error when job directory doesn't exist"""
    # Try to upload tags without starting job (no directory exists)
    try:
        tag_store.upload_tags(sample_tags, "test-job-123")
        assert False
    except Exception:
        pass