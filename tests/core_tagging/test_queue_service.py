import pytest
from dataclasses import replace as dc_replace
import threading
import time
from unittest.mock import Mock

from src.common.errors import BadRequestError, MissingResourceError
from src.service.impl.queue_based import QueueService
from src.service.model import StatusArgs
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs, CompleteJobRequest
from src.api.tagging.request_format import JobSpec, StartJobsRequest
from src.common.content import Content
from src.fetch.model import LiveScope
from tests.core_tagging.conftest import InlineExecutor, enqueue

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
def queue_service(jobstore, dependency_resolver, fake_qfactory) -> QueueService:
    service = QueueService(jobstore, dependency_resolver, arg_resolver=Mock(), qfactory=fake_qfactory)
    service._executor = InlineExecutor() # type: ignore
    return service

def _request(*models: str) -> StartJobsRequest:
    return StartJobsRequest(jobs=[JobSpec(model=m) for m in models])

def test_start_job(queue_service: QueueService, q, make_tag_args):
    enqueue(queue_service, q, [make_tag_args()])
    
    jobs = queue_service.jobstore.list_jobs(ListJobArgs(qid=q.qid), q.token)
    assert len(jobs) == 1
    assert jobs[0].additional_info["title"] == "Test Content Name"

def test_status(queue_service: QueueService, q, make_tag_args):
    job_id = enqueue(queue_service, q, [make_tag_args()])[0]
    job = queue_service.jobstore.get_job(job_id)
    
    assert len(queue_service.status(StatusArgs(qid=q.qid, tenant=None, user=None, title=None))) == 1

    with pytest.raises(MissingResourceError):
        queue_service.status(StatusArgs(qid=q.qid, tenant="iten_other", user=None, title=None))

    assert len(queue_service.status(StatusArgs(qid=q.qid, tenant=job.tenant, user=job.user, title=None))) == 1

def test_job_filter(queue_service: QueueService, q, make_tag_args):
    assert isinstance(queue_service.qfactory, TestQAPIFactory)
    queue_service.qfactory.title = "12 Angry Men"
    enqueue(queue_service, q, [make_tag_args()])

    assert queue_service.status(StatusArgs(qid=q.qid, user=None, tenant=None, title="ANG"))[0].title == "12 Angry Men"
    assert queue_service.status(StatusArgs(qid=q.qid, user=None, tenant=None, title="kong")) == []


def test_live_error_reported_as_cancelled(queue_service: QueueService, make_tag_args, q):
    """Live jobs fail when they end but we should mark them cancelled not failed."""
    live_args = dc_replace(make_tag_args(feature="caption"), scope=LiveScope(stream="video"))
    enqueue(queue_service, q, [make_tag_args(feature="asr"), live_args])

    items = queue_service.jobstore.list_jobs(
        ListJobArgs(qid=q.qid, include_unready=True), q.token
    )
    for item in items:
        assert queue_service.jobstore.claim_job(item.id, auth=q.token)
        queue_service.jobstore.complete_job(
            CompleteJobRequest(id=item.id, status="failed", error="404 Client Error"),
            auth=q.token,
        )

    by_model = {
        r.model: r
        for r in queue_service.status(StatusArgs(qid=q.qid, user=None, tenant=None, title=None))
    }

    assert by_model["caption"].status == "cancelled"
    assert by_model["caption"].error == "404 Client Error"
    assert by_model["asr"].status == "failed"


def test_tag_releases_pending_jobs(queue_service: QueueService, make_tag_args, q):
    queue_service.arg_resolver.resolve.return_value = [make_tag_args(feature="asr")] # type: ignore

    res = queue_service.tag(q, _request("asr"))

    job = queue_service.jobstore.get_job(res[0].job_id)
    assert job.status == "queued"
    assert job.params == make_tag_args(feature="asr")
    assert job.additional_info["title"] == "Test Content Name"


def test_tag_fails_jobs_when_resolve_raises(queue_service: QueueService, q):
    queue_service.arg_resolver.resolve.side_effect = BadRequestError("bad params") # type: ignore

    res = queue_service.tag(q, _request("asr", "caption"))

    for r in res:
        job = queue_service.jobstore.get_job(r.job_id)
        assert job.status == "failed"
        assert job.error == "bad params"


def test_tag_skips_model_already_active(queue_service: QueueService, make_tag_args, q):
    enqueue(queue_service, q, [make_tag_args(feature="asr")])
    queue_service.arg_resolver.resolve = Mock(return_value=[make_tag_args(feature="caption")]) # type: ignore

    caption, asr = queue_service.tag(q, _request("caption", "asr"))

    assert caption.started
    assert not asr.started


def test_tag_model_requested_twice_starts_once(queue_service: QueueService, make_tag_args, q):
    queue_service.arg_resolver.resolve.return_value = [make_tag_args(feature="asr")] # type: ignore

    first, second = queue_service.tag(q, _request("asr", "asr"))

    assert first.started
    assert not second.started
    assert len(queue_service.jobstore.list_jobs(ListJobArgs(qid=q.qid, include_unready=True), q.token)) == 1


def test_stop_while_pending(queue_service: QueueService, make_tag_args, q):
    job_id = queue_service.jobstore.create_job(CreateQueueItem(qid=q.qid, model="asr"), auth=q.token).id

    stop_results = queue_service.stop(q.qid, None)
    queue_service.arg_resolver.resolve.return_value = [make_tag_args(feature="asr")] # type: ignore
    queue_service._resolve_and_release(q, [job_id], _request("asr"))

    assert stop_results[0].job_id == job_id
    job = queue_service.jobstore.get_job(job_id)
    assert job.status == "cancelled"
    assert job.params is None


def test_model_with_pending_request_isnt_started(jobstore, dependency_resolver, fake_qfactory, make_tag_args, q):
    service = QueueService(jobstore, dependency_resolver, arg_resolver=Mock(), qfactory=fake_qfactory)
    finish_resolving = threading.Event()

    def resolve(req, q):
        finish_resolving.wait(5)
        return [make_tag_args(feature=job.model) for job in req.jobs]

    service.arg_resolver.resolve.side_effect = resolve # type: ignore
    first = service.tag(q, _request("asr"))[0]
    other, refused = service.tag(q, _request("caption", "asr"))
    assert not refused.started

    finish_resolving.set()
    time.sleep(0.2)
    assert service.jobstore.get_job(first.job_id).status == "queued"
    assert service.jobstore.get_job(other.job_id).status == "queued"
    assert len(service.jobstore.list_jobs(ListJobArgs(qid=q.qid), q.token)) == 2


def test_pending_job_status(queue_service: QueueService, q):
    queue_service.jobstore.create_job(CreateQueueItem(qid=q.qid, model="asr"), auth=q.token)

    report = queue_service.status(StatusArgs(qid=q.qid, user=None, tenant=None, title=None))[0]

    assert report.status == "pending"
    assert report.model == "asr"
    assert report.params == {}

