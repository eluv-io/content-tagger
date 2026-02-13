
import pytest
from src.tags.tagstore.model import Tag


@pytest.fixture
def job_args(qid):
    """Create a sample job for testing"""
    return {
        "qhit": qid,
        "track": "llava",
        "author": "test-user"
    }


@pytest.fixture
def sample_tags():
    """Create sample tags for testing"""
    return [
        Tag(100, 200, "person", {}, {"confidence": 0.9}, "llava", "test-job-123"),
        Tag(300, 400, "car", {}, {"confidence": 0.8}, "llava", "test-job-123"),
        Tag(500, 600, "hello world", {}, {"language": "en"}, "asr", "test-job-123"),
        Tag(700, 800, "building", {}, {"type": "office"}, "caption", "test-job-123"),
    ]