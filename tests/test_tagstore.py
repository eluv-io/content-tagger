import pytest
import uuid
import os
import json
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore, Tag

def get_random_qid():
    """Generate a random qhit for testing"""
    return f"iq__{uuid.uuid4().hex[:8]}"

@pytest.fixture
def job_args(test_qid):
    """Create a sample job for testing"""
    return {
        "qhit": test_qid,
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
    base_dir = os.path.join(temp_dir, "new_dir")
    assert not os.path.exists(base_dir)

    store = FilesystemTagStore(base_dir=base_dir)
    assert os.path.exists(base_dir)
    assert os.path.isdir(base_dir)


def test_get_job_dir(filesystem_tagstore):
    """Test job directory path generation"""
    job_id = "test-job-123"
    expected_path = os.path.join(filesystem_tagstore.base_path, job_id)
    assert filesystem_tagstore._get_job_dir(job_id) == expected_path


def test_get_job_metadata_path(filesystem_tagstore):
    """Test job metadata path generation"""
    job_id = "test-job-123"
    expected_path = os.path.join(filesystem_tagstore.base_path, job_id, "jobmetadata.json")
    assert filesystem_tagstore._get_job_metadata_path(job_id) == expected_path

def test_get_tags_path(filesystem_tagstore):
    """Test tags file path generation"""
    job_id = "test-job-123"
    source = "llava"
    expected_path = os.path.join(filesystem_tagstore.base_path, job_id, "llava.json")
    assert filesystem_tagstore._get_tags_path(job_id, source) == expected_path


def test_start_job_creates_directory_and_metadata(filesystem_tagstore, job_args):
    """Test that starting a job creates directory and metadata file"""
    tag_store = filesystem_tagstore
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
    assert metadata['track'] == sample_job.track
    assert metadata['author'] == sample_job.author


def test_get_job_returns_correct_job(tag_store, job_args, q):
    """Test retrieving job metadata"""
    sample_job = tag_store.start_job(**job_args, q=q)

    retrieved_job = tag_store.get_job(sample_job.id, q=q)
    
    assert retrieved_job is not None
    assert retrieved_job.id == sample_job.id
    assert retrieved_job.qhit == sample_job.qhit
    assert retrieved_job.track == sample_job.track
    assert retrieved_job.author == sample_job.author


def test_get_job_nonexistent_returns_none(tag_store, q):
    """Test that getting non-existent job returns None"""
    result = tag_store.get_job(str(uuid.uuid4()), q=q)
    assert result is None


def test_upload_tags_creates_source_files(filesystem_tagstore, job_args, sample_tags):
    """Test that uploading tags creates separate files for each source"""
    tag_store = filesystem_tagstore
    sample_job = tag_store.start_job(**job_args)
    tag_store.upload_tags(sample_tags, sample_job.id)
    
    # Check that source files were created
    llava_path = tag_store._get_tags_path(sample_job.id, "llava")
    asr_path = tag_store._get_tags_path(sample_job.id, "asr")
    caption_path = tag_store._get_tags_path(sample_job.id, "caption")
    
    assert os.path.exists(llava_path)
    assert os.path.exists(asr_path)
    assert os.path.exists(caption_path)


def test_upload_tags_without_job_raises_error(tag_store, sample_tags, q):
    """Test that uploading tags without starting job raises error"""
    try:
        tag_store.upload_tags(sample_tags, "test-job-123", q=q)
        assert False
    except Exception:
        pass


def test_upload_tags_appends_to_existing(tag_store, job_args, q):
    """Test that uploading tags appends to existing files"""
    sample_job = tag_store.start_job(**job_args, q=q)
    
    # Upload initial tags
    initial_tags = [
        Tag(100, 200, "person", {}, "llava", sample_job.id),
    ]
    tag_store.upload_tags(initial_tags, sample_job.id, q=q)
    
    # Verify initial upload worked
    llava_tags = tag_store.find_tags(jobid=sample_job.id, q=q)
    assert len(llava_tags) == 1
    assert llava_tags[0].text == "person"
    
    # Upload additional tags
    additional_tags = [
        Tag(300, 400, "car", {}, "llava", sample_job.id),
    ]
    tag_store.upload_tags(additional_tags, sample_job.id, q=q)
    
    # Check that both tags are present
    llava_tags = tag_store.find_tags(jobid=sample_job.id, q=q)
    assert len(llava_tags) == 2
    assert {tag.text for tag in llava_tags} == {"person", "car"}


def test_upload_empty_tags_list(tag_store, job_args, q):
    """Test that uploading empty tags list doesn't create files"""
    sample_job = tag_store.start_job(**job_args, q=q)
    tag_store.upload_tags([], sample_job.id, q=q)

    tags = tag_store.find_tags(jobid=sample_job.id, q=q)
    assert tags == []


def test_get_tags_returns_all_tags(tag_store, job_args, sample_tags, q):
    """Test that find_tags returns tags from all sources"""
    sample_job = tag_store.start_job(**job_args, q=q)
    tag_store.upload_tags(sample_tags, sample_job.id, q=q)
    
    all_tags = tag_store.find_tags(jobid=sample_job.id, q=q)
    
    assert len(all_tags) == 4
    tags = {tag.text for tag in all_tags}
    assert tags == {"person", "car", "hello world", "building"}


def test_get_tags_nonexistent_job_returns_empty(tag_store, q):
    """Test that getting tags for non-existent job returns empty list"""
    tags = tag_store.find_tags(jobid=str(uuid.uuid4()), q=q)
    assert tags == []


def test_find_jobs_no_filters(tag_store, job_args, q, test_qid):
    """Test getting all jobs without filters"""
    sample_job = tag_store.start_job(**job_args, q=q)
    
    # Create another job
    job2 = tag_store.start_job(**{"qhit": test_qid, "stream": "audio", "track": "asr", "author": "user2"}, q=q)
    
    job_ids = tag_store.find_jobs(q=q)
    
    assert set(job_ids) == {sample_job.id, job2.id}


def test_find_jobs_with_qhit_filter(filesystem_tagstore, job_args, q):
    """Test getting jobs filtered by qhit"""
    tag_store = filesystem_tagstore
    sample_job = tag_store.start_job(**job_args, q=q)
    
    # Create job with different qhit
    job2 = tag_store.start_job(**{"qhit": get_random_qid(), "stream": "audio", "track": "asr", "author": "user2"}, q=q)
    job_ids = tag_store.find_jobs(qhit=sample_job.qhit, q=q)
    
    assert job_ids == [sample_job.id]

def test_find_jobs_with_qhit_filter_rest(tag_store, job_args, q):
    """Test getting jobs filtered by qhit
    
    Doesn't create another job with different qhit cause we would need a real content
    """
    sample_job = tag_store.start_job(**job_args, q=q)
    job_ids = tag_store.find_jobs(qhit=sample_job.qhit, q=q)
    
    assert job_ids == [sample_job.id]

def test_find_jobs_with_track_filter(tag_store, job_args, q, test_qid):
    """Test getting jobs filtered by track"""
    sample_job = tag_store.start_job(**job_args, q=q)
    
    # Create job with different track
    tag_store.start_job(**{"qhit": test_qid, "stream": "audio", "track": "asr", "author": "user2"}, q=q)
    
    job_ids = tag_store.find_jobs(track="llava", q=q)
    
    assert job_ids == [sample_job.id]


def test_find_jobs_with_multiple_filters(tag_store, job_args, q, test_qid):
    """Test getting jobs with multiple filters"""
    sample_job = tag_store.start_job(**job_args, q=q)
    
    # Create jobs that match some but not all filters
    job2 = tag_store.start_job(**{"qhit": test_qid, "stream": "audio", "track": "asr", "author": sample_job.author}, q=q)
    job3 = tag_store.start_job(**{"qhit": test_qid, "stream": "video", "track": sample_job.track, "author": "another author"}, q=q)

    job_ids = tag_store.find_jobs(qhit=sample_job.qhit, track=sample_job.track, author=sample_job.author, q=q)

    assert job_ids == [sample_job.id]

def test_upload_tags_handles_missing_job_directory(tag_store, sample_tags, q):
    """Test proper error when job directory doesn't exist"""
    # Try to upload tags without starting job
    try:
        tag_store.upload_tags(sample_tags, str(uuid.uuid4()), q=q)
        assert False
    except Exception:
        pass


def test_start_job_and_upload_tags(tag_store, job_args, sample_tags, q):
    """Test basic job creation and tag upload functionality"""
    sample_job = tag_store.start_job(**job_args, q=q)
    tag_store.upload_tags(sample_tags, sample_job.id, q=q)
    
    # Verify all tags are retrievable
    all_tags = tag_store.find_tags(jobid=sample_job.id, q=q)
    assert len(all_tags) == 4

def test_filter_track(tag_store, q, test_qid):
    """Test filtering by track"""
    job1 = tag_store.start_job(**{"qhit": test_qid, "stream": "video", "track": "llava", "author": "user1"}, q=q)
    job2 = tag_store.start_job(**{"qhit": test_qid, "stream": "audio", "track": "asr", "author": "user2"}, q=q)

    job_ids = tag_store.find_jobs(track="llava", q=q)
    assert job_ids == [job1.id]

    # upload tags
    tags_job1 = [
        Tag(100, 200, "person", {}, "llava", job1.id),
    ]
    tags_job2 = [
        Tag(300, 400, "hello world", {}, "asr", job2.id)
    ]

    tag_store.upload_tags(tags_job1, job1.id, q=q)
    tag_store.upload_tags(tags_job2, job2.id, q=q)

    # filter tags by track
    tags = tag_store.find_tags(track="llava", q=q)
    assert len(tags) == 1
    assert tags[0].text == "person"
    tags = tag_store.find_tags(track="asr", q=q)
    assert len(tags) == 1
    assert tags[0].text == "hello world"


def test_find_tags_basic_filters(tag_store, job_args, sample_tags, q):
    """Test basic tag filtering functionality"""
    sample_job = tag_store.start_job(**job_args, q=q)
    tag_store.upload_tags(sample_tags, sample_job.id, q=q)
    
    # Test filtering by qhit
    tags = tag_store.find_tags(qhit=sample_job.qhit, q=q)
    assert len(tags) == 4
    
    # Test filtering by text content
    tags = tag_store.find_tags(text_contains="hello", q=q)
    assert len(tags) == 1
    assert tags[0].text == "hello world"


def test_find_tags_time_range_filters(tag_store, job_args, sample_tags, q):
    """Test time range filtering"""
    sample_job = tag_store.start_job(**job_args, q=q)
    tag_store.upload_tags(sample_tags, sample_job.id, q=q)
    
    # Test start_time filters
    tags = tag_store.find_tags(start_time_gte=300, q=q)
    assert len(tags) == 3  # Should exclude the first tag (start_time=100)
    
    tags = tag_store.find_tags(start_time_lte=500, q=q)
    assert len(tags) == 3  # Should exclude the last tag (start_time=700)


def test_find_jobs_with_filters(tag_store, q, test_qid):
    """Test job filtering functionality"""
    # Create multiple jobs
    job1 = tag_store.start_job(**{"qhit": test_qid, "stream": "video", "track": "llava", "author": "user1"}, q=q)
    job2 = tag_store.start_job(**{"qhit": test_qid, "stream": "audio", "track": "asr", "author": "user2"}, q=q)
    job3 = tag_store.start_job(**{"qhit": test_qid, "stream": "video", "track": "caption", "author": "user1"}, q=q)

    # Test filtering by author
    job_ids = tag_store.find_jobs(author="user1", q=q)
    assert set(job_ids) == {job1.id, job3.id}

    # Test multiple filters
    job_ids = tag_store.find_jobs(track="caption", author="user1", q=q)
    assert set(job_ids) == {job3.id}

def test_pagination(tag_store, job_args, sample_tags, q):
    """Test pagination functionality"""
    sample_job = tag_store.start_job(**job_args, q=q)
    tag_store.upload_tags(sample_tags, sample_job.id, q=q)
    
    # Test limit
    tags = tag_store.find_tags(limit=2, q=q)
    assert len(tags) == 2
    
    # Test offset
    tags = tag_store.find_tags(offset=2, q=q)
    assert len(tags) == 2
    
    # Test limit + offset
    tags = tag_store.find_tags(limit=1, offset=1, q=q)
    assert len(tags) == 1


def test_count_methods(tag_store, job_args, sample_tags, q):
    """Test counting without loading full data"""
    sample_job = tag_store.start_job(**job_args, q=q)
    tag_store.upload_tags(sample_tags, sample_job.id, q=q)
    
    # Count all tags
    count = tag_store.count_tags(q=q)
    assert count == 4
    
    # Count with filters
    count = tag_store.count_tags(start_time_gte=10, start_time_lte=450, q=q)
    assert count == 2
    
    # Count jobs
    count = tag_store.count_jobs(q=q)
    assert count == 1


def test_error_handling(tag_store, sample_tags, q, test_qid):
    """Test error handling for edge cases"""
    # Test uploading tags without starting job
    with pytest.raises(Exception):
        tag_store.upload_tags(sample_tags, str(uuid.uuid4()), q=q)
    
    # Test empty tags upload
    job = tag_store.start_job(**{"qhit": test_qid, "stream": "video", "track": "track", "author": "user"}, q=q)
    tag_store.upload_tags([], job.id, q=q)  # Should not raise error
    
    # Test getting nonexistent job
    result = tag_store.get_job(str(uuid.uuid4()), q=q)
    assert result is None
    
    # Test getting tags for nonexistent job
    tags = tag_store.find_tags(jobid=str(uuid.uuid4()), q=q)
    assert tags == []


def test_tag_appending(tag_store, job_args, q):
    """Test that tags are properly appended to existing files"""
    sample_job = tag_store.start_job(**job_args, q=q)
    
    # Upload initial tags
    initial_tags = [Tag(100, 200, "person", {}, "llava", sample_job.id)]
    tag_store.upload_tags(initial_tags, sample_job.id, q=q)
    
    # Upload additional tags
    additional_tags = [Tag(300, 400, "car", {}, "llava", sample_job.id)]
    tag_store.upload_tags(additional_tags, sample_job.id, q=q)
    
    # Check that both tags are present
    all_tags = tag_store.find_tags(jobid=sample_job.id, q=q)
    assert len(all_tags) == 2
    assert {tag.text for tag in all_tags} == {"person", "car"}


def test_source_with_slash_encoding(filesystem_tagstore, job_args, q):
    """Test that sources with slashes are properly base64 encoded"""
    tag_store = filesystem_tagstore
    sample_job = tag_store.start_job(**job_args, q=q)
    
    # Create tags with sources containing slashes
    tags_with_slashes = [
        Tag(100, 200, "person", {}, "video/segment_1", sample_job.id),
        Tag(300, 400, "car", {}, "audio/track_2", sample_job.id),
        Tag(500, 600, "building", {}, "normal_source", sample_job.id),  # No slash
    ]
    
    tag_store.upload_tags(tags_with_slashes, sample_job.id, q=q)
    
    # Verify we can retrieve all tags correctly
    all_tags = tag_store.find_tags(jobid=sample_job.id, q=q)
    assert len(all_tags) == 3
    
    # Verify original source names are preserved in the tag data
    tags = {tag.text for tag in all_tags}
    assert tags == {"person", "car", "building"}
    
    # Test filtering by source names with slashes works
    video_tags = tag_store.find_tags(sources=["video/segment_1"], q=q)
    assert len(video_tags) == 1
    assert video_tags[0].text == "person"