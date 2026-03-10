import pytest

from src.service.impl.queue_based import QueueClient
from src.tagging.fabric_tagging.queue.model import ListJobArgs

class TestContent:
    def __init__(self, qid: str):
        self.qid = qid

    def content_object_metadata(self, metadata_subtree: str):
        if metadata_subtree == "/public/name":
            return "Test Content Name"
        return None

    def token(self) -> str:
        return ""

@pytest.fixture
def queue_service(jobstore) -> QueueClient:
    return QueueClient(jobstore)

@pytest.fixture
def content():
    return TestContent("test")

def test_start_job(queue_service: QueueClient, content, make_tag_args):
    args = make_tag_args()
    result = queue_service.tag(content, args)
    assert result.started
    assert result.job_id != ""
    
    jobs = queue_service.jobstore.list_jobs(ListJobArgs(qid=content.qid), content.token())
    assert len(jobs) == 1
    assert jobs[0].additional_info["title"] == "Test Content Name"
