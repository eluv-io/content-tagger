"""Tests for QueueManagerJobStore's mapping onto the queue manager API, against a mocked session."""

from unittest.mock import Mock

import pytest

from src.common.errors import BadRequestError, MissingResourceError
from src.fetch.model import VideoScope
from src.service.model import TagDetails
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.model import CompleteJobRequest, CreateQueueItem, ListJobArgs, ReleaseJobRequest
from src.tagging.fabric_tagging.queue.qmanager_jobstore import PAGE_SIZE, QueueManagerJobStore


def _job(id: str = "j1", **fields) -> dict:
    return {
        "id": id,
        "qid": "iq__test",
        "type": "tag",
        "subtype": "caption",
        "status": "queued",
        "created_at": "2026-10-07T12:00:00Z",
        "params": None,
        "additional_info": {},
        "user": "0xabc",
        "tenant": "iten1",
        **fields,
    }


def _args() -> TagArgs:
    return TagArgs(
        feature="caption", run_config={}, scope=VideoScope(stream="video"), track_suffix="", replace=False,
        destination_qid="", max_fetch_retries=3, caller_info={}, index_qid="",
    )


def _response(status: int = 200, body: dict | None = None) -> Mock:
    return Mock(status_code=status, ok=status < 400, json=Mock(return_value=body), text=str(body))


@pytest.fixture
def store() -> QueueManagerJobStore:
    store = QueueManagerJobStore("http://qm/", worker_secret="secret", timeout=5)
    store.session.request = Mock()
    return store


def _calls(store) -> list[tuple]:
    return [(c.args[0], c.args[1]) for c in store.session.request.call_args_list]


def test_worker_calls_carry_only_the_secret(store):
    store.session.request.return_value = _response(200, _job(status="running"))
    store.claim_job("j1", auth="tok")
    assert store.session.request.call_args.kwargs["headers"] == {"X-Worker-Secret": "secret"}


def test_user_calls_carry_only_the_token(store):
    # the worker secret would take precedence, and workers can't cancel
    store.session.request.return_value = _response(200, _job(status="cancelling"))
    store.cancel_job("j1", auth="tok")
    assert store.session.request.call_args.kwargs["headers"] == {"Authorization": "Bearer tok"}


def test_create_job_is_held(store):
    store.session.request.return_value = _response(201, _job(status="pending"))

    item = store.create_job(CreateQueueItem(qid="iq__test", model="caption"), auth="tok")

    call = store.session.request.call_args
    assert call.args[:2] == ("POST", "http://qm/q/iq__test/jobs")
    assert call.kwargs["json"] == {"type": "tag", "subtype": "caption", "hold": True}
    assert call.kwargs["headers"] == {"Authorization": "Bearer tok"}
    assert item.status == "pending"
    assert item.model == "caption"
    assert item.params is None
    assert item.auth == "tok"
    assert item.created_at == 1791374400.0


def test_release_sets_resource_then_releases_with_params(store):
    store.session.request.return_value = _response(200, _job())

    assert store.release_job(ReleaseJobRequest(id="j1", params=_args(), deps=["j0"], additional_info={"title": "t"}, resource="iq__test/caption/video"), auth="tok")

    assert _calls(store) == [("PATCH", "http://qm/worker/jobs/j1"), ("POST", "http://qm/jobs/j1/release")]
    assert store.session.request.call_args_list[0].kwargs["json"] == {"resource_hash": "iq__test/caption/video"}
    release = store.session.request.call_args_list[1].kwargs["json"]
    assert release["dependencies"] == ["j0"]
    assert release["additional_info"] == {"title": "t", "dependencies": ["j0"]}
    assert release["params"]["scope"]["type"] == "video"
    assert release["enforce_resource"] == "active"


def test_release_refused_when_not_pending(store):
    store.session.request.return_value = _response(409, {"error": "not pending"})
    assert store.release_job(ReleaseJobRequest(id="j1", params=_args(), deps=[], additional_info={}, resource="r"), auth="tok") is False


def test_job_conversion(store):
    params = {
        "feature": "caption", "run_config": {}, "scope": {"type": "video", "stream": "video"}, "replace": False,
        "destination_qid": "", "max_fetch_retries": 3,
    }
    store.session.request.return_value = _response(200, _job(
        params=params,
        additional_info={"title": "t", "dependencies": ["j0"]},
        status="cancelled",
        status_details={"system_message": "parent job j0 cancelled"},
    ))

    item = store.get_job("j1")

    assert item.params.scope == VideoScope(stream="video")
    assert item.deps == ["j0"]
    assert item.additional_info == {"title": "t"}
    assert item.status_details is None
    assert item.error == "parent job j0 cancelled"


def test_auth_comes_from_stored_authorization_header(store):
    store.session.request.return_value = _response(200, _job(auth={"Authorization": "Bearer tok"}))
    assert store.get_job("j1").auth == "tok"


def test_system_message_on_succeeded_job_isnt_an_error(store):
    store.session.request.return_value = _response(200, _job(
        status="succeeded", status_details={"system_message": "marked to be cancelled but finished instead"},
    ))
    assert store.get_job("j1").error is None


def test_archived_job_is_missing(store):
    store.session.request.return_value = _response(200, _job(status="succeeded", archived=True))
    with pytest.raises(MissingResourceError):
        store.get_job("j1")


def test_delete(store):
    store.session.request.return_value = _response(200, _job(status="succeeded", archived=True))

    store.delete_job("j1", auth="tok")

    call = store.session.request.call_args
    assert call.args[:2] == ("DELETE", "http://qm/jobs/j1")
    assert call.kwargs["headers"] == {"Authorization": "Bearer tok"}


def test_delete_unended_job(store):
    store.session.request.return_value = _response(409, {"error": "job hasn't ended"})
    with pytest.raises(BadRequestError):
        store.delete_job("j1", auth="tok")


def test_missing_job(store):
    store.session.request.return_value = _response(404, {"error": "not found"})
    with pytest.raises(MissingResourceError):
        store.get_job("nope")


def test_list_pages_through_results(store):
    first = [_job(f"a{i}") for i in range(PAGE_SIZE)]
    store.session.request.side_effect = [
        _response(200, {"jobs": first, "meta": {"total": PAGE_SIZE + 1}}),
        _response(200, {"jobs": [_job("last")], "meta": {"total": PAGE_SIZE + 1}}),
    ]

    items = store.list_jobs(ListJobArgs(qid="iq__test"), auth="")

    assert len(items) == PAGE_SIZE + 1
    params = [c.kwargs["params"] for c in store.session.request.call_args_list]
    assert params[0] == {
        "type": "tag", "qid": "iq__test", "include_unready": "false", "ignore_resource_hash": "false", "start": 0, "limit": PAGE_SIZE,
    }
    assert params[1]["start"] == PAGE_SIZE


def test_list_limit_is_one_page(store):
    store.session.request.return_value = _response(200, {"jobs": [_job("a"), _job("b")], "meta": {"total": 10}})

    items = store.list_jobs(ListJobArgs(status="queued", limit=2), auth="")

    assert len(items) == 2
    assert store.session.request.call_count == 1
    assert store.session.request.call_args.kwargs["params"]["limit"] == 2


def test_claim_refused(store):
    store.session.request.return_value = _response(400, {"error": "job not in queued state"})
    assert store.claim_job("j1", auth="") is False


def _details() -> TagDetails:
    return TagDetails(
        tag_status="Tagging content", time_running=1.0, progress=0.5, tagging_progress="1/2", tagged_duration=0,
        total_parts=2, downloaded_parts=2, tagged_parts=1, warnings=None,
    )


def test_update_progress_returns_job(store):
    store.session.request.return_value = _response(200, _job(status="cancelling"))
    assert store.update_progress("j1", _details(), auth="").status == "cancelling"
    assert store.session.request.call_args.kwargs["json"]["status_details"]["progress"] == 0.5


def test_update_progress_on_ended_job(store):
    store.session.request.side_effect = [_response(409, {"error": "ended"}), _response(200, _job(status="succeeded"))]
    assert store.update_progress("j1", _details(), auth="").status == "succeeded"


def test_complete_refused(store):
    store.session.request.return_value = _response(409, {"error": "job not running"})
    assert store.complete_job(CompleteJobRequest(id="j1", status="succeeded"), auth="") is False
    assert store.session.request.call_args.kwargs["json"] == {"status": "succeeded"}


def test_cancel_records_reason(store):
    store.session.request.side_effect = [_response(200, _job(status="cancelled")), _response(200, _job(status="cancelled"))]

    store.cancel_job("j1", auth="tok", reason="duplicate")

    assert _calls(store) == [("POST", "http://qm/jobs/j1/cancel"), ("POST", "http://qm/worker/jobs/j1/complete")]
    assert store.session.request.call_args.kwargs["json"] == {"status_details": {"system_message": "duplicate"}}


def test_cancel_ended_job_is_a_noop(store):
    store.session.request.return_value = _response(409, {"error": "already ended"})
    store.cancel_job("j1", auth="tok", reason="duplicate")
    assert store.session.request.call_count == 1
