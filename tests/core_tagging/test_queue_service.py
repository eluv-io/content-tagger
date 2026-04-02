import pytest
from unittest.mock import Mock

from src.common.errors import MissingResourceError
from src.service.impl.queue_based import QueueService
from src.service.model import StatusArgs
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs
from src.common.content import Content

class TestQAPIFactory:
    def __init__(self):
        self.title = "Test Content Name"

    def create(self, q: Content):
        return Mock(
            content_object_metadata=Mock(return_value=self.title),
            id=Mock(return_value=q.qid),
            token=Mock(return_value=q.token)
        )
    
@pytest.fixture
def fake_qfactory():
    return TestQAPIFactory()

@pytest.fixture
def queue_service(jobstore, fake_qfactory) -> QueueService:
    return QueueService(jobstore, fake_qfactory)

def test_start_job(queue_service: QueueService, make_tag_args):
    args = make_tag_args()
    content = Content(qid="test", token="")
    result = queue_service.tag(content, args)
    assert result.started
    assert result.job_id != ""
    
    jobs = queue_service.jobstore.list_jobs(ListJobArgs(qid=content.qid), content.token)
    assert len(jobs) == 1
    assert jobs[0].additional_info["title"] == "Test Content Name"

def test_status(queue_service: QueueService, make_tag_args):
    args = make_tag_args()
    content = Content(qid="test", token="")
    queue_service.tag(content, args)
    
    status_results = queue_service.status(StatusArgs(
        qid=None,
        tenant=None,
        user=None,
        title=None,
    ))
    
    assert len(status_results) == 1

    with pytest.raises(MissingResourceError):
        queue_service.status(StatusArgs(
            qid=content.qid,
            tenant="something else",
            user=None,
            title=None,
        ))

    status_results = queue_service.status(StatusArgs(
        tenant=None,
        user="0x123",
        title=None,
        qid=None,
    ))

    assert len(status_results) == 1

def test_job_filter(queue_service: QueueService, make_tag_args):
    assert isinstance(queue_service.qfactory, TestQAPIFactory)
    queue_service.qfactory.title = "12 Angry Men"

    args = make_tag_args()
    content = Content(qid="test", token="")
    res = queue_service.tag(content, args)
    assert res.started

    content = Content(qid="test2", token="")
    queue_service.qfactory.title = "King Kong"
    res = queue_service.tag(content, args)
    assert res.started
    assert queue_service.status(StatusArgs(
        qid=None,
        user=None,
        tenant=None,
        title="kin"
    ))[0].title == "King Kong"

    assert queue_service.status(StatusArgs(
        qid=None,
        user=None,
        tenant=None,
        title="ang"
    ))[0].title == "12 Angry Men"

