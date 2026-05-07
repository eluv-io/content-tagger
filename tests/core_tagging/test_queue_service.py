import pytest
from unittest.mock import Mock

from src.common.errors import MissingResourceError
from src.service.impl.queue_based import QueueService
from src.service.job_poster import JobPoster
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
def job_poster(queue_jobstore, track_resolver, fake_qfactory, model_configs) -> JobPoster:
    """Create a JobPoster for testing, using the queue_jobstore and other dependencies."""
    return JobPoster(
        job_store=queue_jobstore,
        track_resolver=track_resolver,
        model_configs=model_configs,
        qfactory=fake_qfactory
    )

@pytest.fixture
def queue_service(job_poster) -> QueueService:
    return QueueService(job_poster)

def test_start_job(queue_service: QueueService, make_tag_args):
    args = make_tag_args()
    content = Content(qid="test", token="")
    result = queue_service.tag(content, [args])[0]
    assert result.started
    assert result.job_id != ""
    
    jobs = queue_service.jobstore.list_jobs(ListJobArgs(qid=content.qid), content.token)
    assert len(jobs) == 1
    assert jobs[0].additional_info["title"] == "Test Content Name"

def test_status(queue_service: QueueService, make_tag_args):
    args = make_tag_args()
    content = Content(qid="test", token="")
    queue_service.tag(content, [args])
    
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
    assert isinstance(queue_service.job_poster.qfactory, TestQAPIFactory)
    queue_service.job_poster.qfactory.title = "12 Angry Men"

    args = make_tag_args()
    content = Content(qid="test", token="")
    res = queue_service.tag(content, [args])[0]
    assert res.started

    content = Content(qid="test2", token="")
    queue_service.job_poster.qfactory.title = "King Kong"
    res = queue_service.tag(content, [args])[0]
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
        title="ANG"
    ))[0].title == "12 Angry Men"

