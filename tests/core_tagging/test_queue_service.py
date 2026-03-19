import pytest
from unittest.mock import Mock

from src.service.impl.queue_based import QueueClient
from src.tagging.fabric_tagging.queue.model import ListJobArgs
from src.common.content import Content

class TestQAPIFactory:
    def create(self, q: Content):
        return Mock(
            content_object_metadata=Mock(return_value="Test Content Name"),
            id=Mock(return_value=q.qid),
            token=Mock(return_value=q.token)
        )
    
@pytest.fixture
def fake_qfactory():
    return TestQAPIFactory()

@pytest.fixture
def queue_service(jobstore, fake_qfactory) -> QueueClient:
    return QueueClient(jobstore, fake_qfactory)

def test_start_job(queue_service: QueueClient, make_tag_args):
    args = make_tag_args()
    content = Content(qid="test", token="")
    result = queue_service.tag(content, args)
    assert result.started
    assert result.job_id != ""
    
    jobs = queue_service.jobstore.list_jobs(ListJobArgs(qid=content.qid), content.token)
    assert len(jobs) == 1
    assert jobs[0].additional_info["title"] == "Test Content Name"
