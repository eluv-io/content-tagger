import pytest
from concurrent.futures import Executor, Future
from dataclasses import replace as dc_replace
import threading
import time
from unittest.mock import Mock

from src.common.errors import BadRequestError, MissingResourceError
from src.service.dependency_resolver import JobDependencies
from src.service.impl.queue_based import QueueService
from src.service.model import StatusArgs
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs, CompleteJobRequest
from src.api.tagging.request_format import JobSpec, StartJobsRequest
from src.common.content import Content
from src.fetch.model import LiveScope
from tests.core_tagging.conftest import enqueue

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

class InlineExecutor(Executor):
    """Runs submitted work immediately so background releases are deterministic in tests."""
    def submit(self, fn, /, *args, **kwargs):
        f = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as e:
            f.set_exception(e)
        return f

@pytest.fixture
def queue_service(queue_jobstore, dependency_resolver, fake_qfactory) -> QueueService:
    service = QueueService(queue_jobstore, dependency_resolver, arg_resolver=Mock(), qfactory=fake_qfactory)
    service._executor = InlineExecutor() # type: ignore
    return service

def _request(*models: str) -> StartJobsRequest:
    return StartJobsRequest(jobs=[JobSpec(model=m) for m in models])

def test_start_job(queue_service: QueueService, make_tag_args):
    args = make_tag_args()
    content = Content(qid="test", token="")
    enqueue(queue_service, content, [args])
    
    jobs = queue_service.jobstore.list_jobs(ListJobArgs(qid=content.qid), content.token)
    assert len(jobs) == 1
    assert jobs[0].additional_info["title"] == "Test Content Name"

def test_status(queue_service: QueueService, make_tag_args):
    args = make_tag_args()
    content = Content(qid="test", token="")
    enqueue(queue_service, content, [args])
    
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
    enqueue(queue_service, content, [args])

    content = Content(qid="test2", token="")
    queue_service.qfactory.title = "King Kong"
    enqueue(queue_service, content, [args])
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


def test_live_error_reported_as_cancelled(queue_service: QueueService, make_tag_args):
    """An errored live job reports 'cancelled' but keeps its error; vod still reports 'failed'."""
    content = Content(qid="test", token="")
    live_args = dc_replace(make_tag_args(feature="caption"), scope=LiveScope(stream="video"))
    enqueue(queue_service, content, [make_tag_args(feature="asr"), live_args])

    items = queue_service.jobstore.list_jobs(
        ListJobArgs(qid=content.qid, include_unready=True), content.token
    )
    for item in items:
        assert queue_service.jobstore.claim_job(item.id, auth=content.token)
        queue_service.jobstore.complete_job(
            CompleteJobRequest(id=item.id, status="failed", error="404 Client Error"),
            auth=content.token,
        )

    by_model = {
        r.model: r
        for r in queue_service.status(StatusArgs(qid=content.qid, user=None, tenant=None, title=None))
    }

    assert by_model["caption"].status == "cancelled"
    assert by_model["caption"].error == "404 Client Error"
    assert by_model["asr"].status == "failed"


def test_tag_releases_pending_jobs(queue_service: QueueService, make_tag_args):
    content = Content(qid="test", token="")
    queue_service.arg_resolver.resolve.return_value = [make_tag_args(feature="asr")] # type: ignore

    res = queue_service.tag(content, _request("asr"))

    job = queue_service.jobstore.get_job(res[0].job_id)
    assert job.status == "queued"
    assert job.params == make_tag_args(feature="asr")
    assert job.additional_info["title"] == "Test Content Name"


def test_tag_fails_jobs_when_resolve_raises(queue_service: QueueService):
    content = Content(qid="test", token="")
    queue_service.arg_resolver.resolve.side_effect = BadRequestError("bad params") # type: ignore

    res = queue_service.tag(content, _request("asr", "caption"))

    for r in res:
        job = queue_service.jobstore.get_job(r.job_id)
        assert job.status == "failed"
        assert job.error == "bad params"


def test_tag_cancels_duplicate(queue_service: QueueService, make_tag_args):
    content = Content(qid="test", token="")
    first_id = enqueue(queue_service, content, [make_tag_args(feature="asr")])[0]
    dup_id = enqueue(queue_service, content, [make_tag_args(feature="asr")])[0]

    job = queue_service.jobstore.get_job(dup_id)
    assert job.status == "cancelled"
    assert first_id in (job.error or "")


def test_stop_while_pending(queue_service: QueueService, make_tag_args):
    content = Content(qid="test", token="")
    job_id = queue_service.jobstore.create_job(CreateQueueItem(qid=content.qid, model="asr"), auth="").id

    stop_results = queue_service.stop(content.qid, None)
    queue_service.release(content, [job_id], [make_tag_args(feature="asr")])

    assert stop_results[0].job_id == job_id
    job = queue_service.jobstore.get_job(job_id)
    assert job.status == "cancelled"
    assert job.params is None


def test_refuses_model_with_pending_request(queue_jobstore, dependency_resolver, fake_qfactory, make_tag_args):
    service = QueueService(queue_jobstore, dependency_resolver, arg_resolver=Mock(), qfactory=fake_qfactory)
    content = Content(qid="test", token="")
    finish_resolving = threading.Event()

    def resolve(req, q):
        finish_resolving.wait(5)
        return [make_tag_args(feature=job.model) for job in req.jobs]

    service.arg_resolver.resolve.side_effect = resolve # type: ignore
    first = service.tag(content, _request("asr"))[0]
    with pytest.raises(BadRequestError):
        service.tag(content, _request("caption", "asr"))
    other = service.tag(content, _request("caption"))[0]

    finish_resolving.set()
    time.sleep(0.2)
    assert service.jobstore.get_job(first.job_id).status == "queued"
    assert service.jobstore.get_job(other.job_id).status == "queued"
    # the refused request wrote no jobs
    assert len(service.jobstore.list_jobs(ListJobArgs(qid="test"), "")) == 2


def test_pending_job_status(queue_service: QueueService):
    content = Content(qid="test", token="")
    queue_service.jobstore.create_job(CreateQueueItem(qid="test", model="asr"), auth="")

    report = queue_service.status(StatusArgs(qid="test", user=None, tenant=None, title=None))[0]

    assert report.status == "pending"
    assert report.model == "asr"
    assert report.params == {}


def test_release_refused_by_queue_cancels_job(queue_service: QueueService, queue_jobstore, fake_qfactory, make_tag_args):
    """A job active for the same model and stream that local duplicate detection missed, e.g. submitted through
    another tagger instance, makes the queue refuse the release."""
    content = Content(qid="test", token="")
    args = make_tag_args(feature="caption", stream="video")
    first_id = enqueue(queue_service, content, [args])[0]

    resolver = Mock(resolve=Mock(return_value=[JobDependencies(parents=[], duplicate_of=None)]))
    other_instance = QueueService(queue_jobstore, resolver, arg_resolver=Mock(), qfactory=fake_qfactory)
    second_id = enqueue(other_instance, content, [args])[0]

    assert queue_jobstore.get_job(first_id).status == "queued"
    second = queue_jobstore.get_job(second_id)
    assert second.status == "cancelled"
    assert "already active" in second.error
