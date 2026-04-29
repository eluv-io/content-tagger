
import pytest
import uuid
from src.tags.tagstore.model import Tag

def make_tag(start_time, end_time, text, additional_info, source, batch_id, frame_info=None):
    return Tag(
        id=str(uuid.uuid4()),
        start_time=start_time,
        end_time=end_time,
        text=text,
        additional_info=additional_info,
        source=source,
        batch_id=batch_id,
        frame_info=frame_info,
    )

@pytest.fixture
def job_args():
    """Create a sample job for testing"""
    return {
        "track": "llava",
        "author": "test-user"
    }


@pytest.fixture
def sample_tags():
    """Create sample tags for testing"""
    return [
        make_tag(100, 200, "person", None, "llava", "test-job-123"),
        make_tag(300, 400, "car", None, "llava", "test-job-123"),
        make_tag(500, 600, "hello world", None, "asr", "test-job-123"),
        make_tag(700, 800, "building", None, "caption", "test-job-123"),
    ]