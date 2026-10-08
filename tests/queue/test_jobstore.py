"""Tests for the JobStore interface, exercised via the jobstore fixture. Jobs are created on the q fixture's content
with its token, so the tests also run against a real queue manager."""

import uuid

import pytest

from src.common.content import Content
from src.common.errors import JobConflictError, MissingResourceError
from src.fetch.model import VideoScope
from src.service.model import TagDetails
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import (
    CompleteJobRequest,
    CreateQueueItem,
    ListJobArgs,
    ReleaseJobRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tag_args(feature: str = "test_feature") -> TagArgs:
    return TagArgs(
        feature=feature,
        run_config={},
        scope=VideoScope(stream="video", start_time=0, end_time=60),
        replace=False,
        destination_qid="iq__dest",
        index_qid="iq__index",
        track_suffix="",
        caller_info={},
        max_fetch_retries=3,
    )


def _unique_model() -> str:
    # a model runs at most once at a time on a content object, so unrelated jobs need their own
    return f"model_{uuid.uuid4().hex[:8]}"


def _create_pending(jobstore: JobStore, q: Content, feature: str = "test_feature"):
    return jobstore.create_job(CreateQueueItem(qid=q.qid, model=feature), auth=q.token)


def _create_queued(jobstore: JobStore, q: Content, feature: str | None = None, deps: list = [], additional_info: dict = {}):
    feature = feature or _unique_model()
    job = _create_pending(jobstore, q, feature)
    jobstore.release_job(
        ReleaseJobRequest(id=job.id, params=_make_tag_args(feature), deps=deps, additional_info=additional_info),
        auth=q.token,
    )
    return jobstore.get_job(job.id)


def _create_running(jobstore: JobStore, q: Content, **kwargs):
    job = _create_queued(jobstore, q, **kwargs)
    assert jobstore.claim_job(job.id, auth=q.token)
    return job


def _details(progress: float = 0.5) -> TagDetails:
    return TagDetails(
        tag_status="Tagging content",
        time_running=1.0,
        progress=progress,
        tagging_progress="1/2",
        tagged_duration=0,
        total_parts=2,
        downloaded_parts=2,
        tagged_parts=1,
        warnings=None,
    )


def _list(jobstore: JobStore, q: Content, **kwargs) -> list:
    return jobstore.list_jobs(ListJobArgs(qid=q.qid, **kwargs), auth=q.token)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateAndList:
    def test_create_job_appears_in_list(self, jobstore, q):
        _create_queued(jobstore, q)
        assert len(_list(jobstore, q)) == 1

    def test_created_job_has_correct_qid(self, jobstore, q):
        _create_queued(jobstore, q)
        assert _list(jobstore, q)[0].qid == q.qid

    def test_created_job_has_correct_feature(self, jobstore, q):
        _create_queued(jobstore, q, feature="my_feature")
        jobs = _list(jobstore, q)
        assert jobs[0].model == "my_feature"
        assert jobs[0].params.feature == "my_feature"

    def test_released_job_is_queued(self, jobstore, q):
        _create_queued(jobstore, q)
        assert len(_list(jobstore, q, status="queued")) == 1

    def test_multiple_jobs_all_listed(self, jobstore, q):
        _create_queued(jobstore, q)
        _create_queued(jobstore, q)
        assert len(_list(jobstore, q)) == 2

    def test_additional_info_is_stored_and_retrieved(self, jobstore, q):
        info = {"key1": "value1", "key2": 42}
        _create_queued(jobstore, q, additional_info=info)
        jobs = _list(jobstore, q)
        assert len(jobs) == 1
        assert jobs[0].additional_info == info

    def test_job_runs_with_the_submitters_token(self, jobstore, q):
        _create_queued(jobstore, q)
        assert _list(jobstore, q)[0].auth == q.token


class TestListFiltering:
    def test_filter_by_qid(self, jobstore, q):
        _create_queued(jobstore, q)
        assert jobstore.list_jobs(ListJobArgs(qid="iq__nojobs"), auth=q.token) == []

    def test_filter_by_status_returns_empty_when_no_match(self, jobstore, q):
        _create_queued(jobstore, q)
        assert _list(jobstore, q, status="running") == []

    def test_filter_by_status_after_claim(self, jobstore, q):
        job = _create_queued(jobstore, q)
        jobstore.claim_job(job.id, auth=q.token)

        running = _list(jobstore, q, status="running")
        assert len(running) == 1
        assert running[0].id == job.id


class TestClaimJob:
    def test_claim_queued_job_returns_true(self, jobstore, q):
        job = _create_queued(jobstore, q)
        assert jobstore.claim_job(job.id, auth=q.token) is True

    def test_claim_moves_job_to_running(self, jobstore, q):
        job = _create_queued(jobstore, q)
        jobstore.claim_job(job.id, auth=q.token)

        assert len(_list(jobstore, q, status="queued")) == 0
        assert len(_list(jobstore, q, status="running")) == 1

    def test_claim_already_running_job_returns_false(self, jobstore, q):
        job = _create_queued(jobstore, q)
        jobstore.claim_job(job.id, auth=q.token)
        assert jobstore.claim_job(job.id, auth=q.token) is False


class TestCompleteJob:
    def test_complete_running_job_as_succeeded(self, jobstore, q):
        job = _create_running(jobstore, q)

        assert jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth=q.token)

        succeeded = _list(jobstore, q, status="succeeded")
        assert len(succeeded) == 1
        assert succeeded[0].id == job.id

    def test_complete_running_job_as_failed_with_error(self, jobstore, q):
        job = _create_running(jobstore, q)

        jobstore.complete_job(
            CompleteJobRequest(id=job.id, status="failed", error="something went wrong", status_details=_details()),
            auth=q.token,
        )

        failed = _list(jobstore, q, status="failed")
        assert len(failed) == 1
        assert failed[0].error == "something went wrong"
        assert failed[0].status_details == _details()

    def test_queued_job_cant_complete(self, jobstore, q):
        job = _create_queued(jobstore, q)
        assert not jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth=q.token)

    def test_ended_job_cant_complete(self, jobstore, q):
        job = _create_running(jobstore, q)
        jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth=q.token)
        assert not jobstore.complete_job(CompleteJobRequest(id=job.id, status="failed"), auth=q.token)
        assert jobstore.get_job(job.id).status == "succeeded"

    def test_failure_cascades_to_dependents(self, jobstore, q):
        parent = _create_running(jobstore, q)
        child = _create_queued(jobstore, q, deps=[parent.id])
        grandchild = _create_queued(jobstore, q, deps=[child.id])

        jobstore.complete_job(CompleteJobRequest(id=parent.id, status="failed"), auth=q.token)

        assert jobstore.get_job(child.id).status == "failed"
        assert jobstore.get_job(grandchild.id).status == "failed"


class TestProgress:
    def test_update_progress_records_details(self, jobstore, q):
        job = _create_running(jobstore, q)
        job = jobstore.update_progress(job.id, _details(0.25), auth=q.token)
        assert job.status == "running"
        assert job.status_details == _details(0.25)

    def test_update_progress_reports_stop_request(self, jobstore, q):
        job = _create_running(jobstore, q)
        jobstore.cancel_job(job.id, auth=q.token)
        assert jobstore.update_progress(job.id, _details(), auth=q.token).status == "cancelling"

    def test_update_progress_on_ended_job_returns_status(self, jobstore, q):
        job = _create_running(jobstore, q)
        jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth=q.token)
        assert jobstore.update_progress(job.id, _details(), auth=q.token).status == "succeeded"


class TestStopJob:
    def test_cancelling_job_completes_as_cancelled(self, jobstore, q):
        job = _create_running(jobstore, q)
        jobstore.cancel_job(job.id, auth=q.token)
        assert jobstore.complete_job(CompleteJobRequest(id=job.id, status="cancelled"), auth=q.token)
        assert jobstore.get_job(job.id).status == "cancelled"

    def test_cancelling_job_that_ended_on_its_own_completes_as_succeeded(self, jobstore, q):
        job = _create_running(jobstore, q)
        jobstore.cancel_job(job.id, auth=q.token)
        assert jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth=q.token)
        job = jobstore.get_job(job.id)
        assert job.status == "succeeded"
        assert job.error is None


def test_dependency_listing(jobstore: JobStore, q):
    job = _create_queued(jobstore, q)
    _create_queued(jobstore, q, deps=[job.id])
    jobs = _list(jobstore, q)
    assert len(jobs) == 1
    assert jobs[0].id == job.id
    assert len(_list(jobstore, q, include_unready=True)) == 2


class TestPendingJobs:
    def test_pending_job_is_not_claimable(self, jobstore, q):
        job = _create_pending(jobstore, q, feature="m")
        assert job.status == "pending"
        assert job.model == "m"
        assert job.params is None
        assert _list(jobstore, q, status="queued") == []
        assert jobstore.claim_job(job.id, auth=q.token) is False

    def test_release_sets_params_and_queues(self, jobstore, q):
        dep = _create_queued(jobstore, q)
        job = _create_pending(jobstore, q, feature="m")

        released = jobstore.release_job(
            ReleaseJobRequest(id=job.id, params=_make_tag_args("m"), deps=[dep.id], additional_info={"title": "t"}),
            auth=q.token,
        )

        assert released is True
        job = jobstore.get_job(job.id)
        assert job.status == "queued"
        assert job.params == _make_tag_args("m")
        assert job.additional_info == {"title": "t"}
        # waits on dep
        assert jobstore.claim_job(job.id, auth=q.token) is False

    def test_release_fails_if_not_pending(self, jobstore, q):
        job = _create_pending(jobstore, q, feature="m")
        jobstore.cancel_job(job.id, auth=q.token)

        released = jobstore.release_job(
            ReleaseJobRequest(id=job.id, params=_make_tag_args("m"), deps=[], additional_info={}),
            auth=q.token,
        )

        assert released is False
        assert jobstore.get_job(job.id).status == "cancelled"

    def test_pending_dependency_blocks_child(self, jobstore, q):
        parent = _create_pending(jobstore, q, feature="m")
        _create_queued(jobstore, q, deps=[parent.id])
        assert _list(jobstore, q, status="queued") == []


class TestStopCancels:
    def test_stop_pending_job_cancels(self, jobstore, q):
        job = _create_pending(jobstore, q, feature="m")
        jobstore.cancel_job(job.id, auth=q.token, reason="duplicate")
        job = jobstore.get_job(job.id)
        assert job.status == "cancelled"
        assert job.error == "duplicate"

    def test_stop_queued_job_cancels(self, jobstore, q):
        job = _create_queued(jobstore, q)
        jobstore.cancel_job(job.id, auth=q.token)
        assert jobstore.get_job(job.id).status == "cancelled"
        assert jobstore.claim_job(job.id, auth=q.token) is False

    def test_stop_running_job_makes_it_cancelling(self, jobstore, q):
        job = _create_running(jobstore, q)
        jobstore.cancel_job(job.id, auth=q.token)
        assert jobstore.get_job(job.id).status == "cancelling"

    def test_cancel_cascades_to_dependents(self, jobstore, q):
        parent = _create_running(jobstore, q)
        child = _create_queued(jobstore, q, deps=[parent.id])

        jobstore.cancel_job(parent.id, auth=q.token)

        child = jobstore.get_job(child.id)
        assert child.status == "cancelled"
        assert parent.id in child.error


class TestDependencies:
    def test_child_is_claimable_once_parent_succeeds(self, jobstore, q):
        parent = _create_running(jobstore, q)
        child = _create_queued(jobstore, q, deps=[parent.id])
        assert jobstore.claim_job(child.id, auth=q.token) is False

        jobstore.complete_job(CompleteJobRequest(id=parent.id, status="succeeded"), auth=q.token)

        assert [j.id for j in _list(jobstore, q, status="queued")] == [child.id]
        assert jobstore.claim_job(child.id, auth=q.token) is True


def test_list_limit(jobstore, q):
    for _ in range(3):
        _create_queued(jobstore, q)
    assert len(_list(jobstore, q, status="queued", limit=2)) == 2


def test_delete_ended_job(jobstore, q):
    job = _create_running(jobstore, q)
    jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth=q.token)

    jobstore.delete_job(job.id, auth=q.token)

    with pytest.raises(MissingResourceError):
        jobstore.get_job(job.id)
    assert _list(jobstore, q) == []


class TestResources:
    def test_create_refused_while_model_active_on_content(self, jobstore, q):
        _create_pending(jobstore, q, feature="m")
        with pytest.raises(JobConflictError):
            _create_pending(jobstore, q, feature="m")

    def test_create_allowed_once_model_job_ended(self, jobstore, q):
        first = _create_running(jobstore, q, feature="m")
        jobstore.complete_job(CompleteJobRequest(id=first.id, status="succeeded"), auth=q.token)
        assert _create_queued(jobstore, q, feature="m").status == "queued"
