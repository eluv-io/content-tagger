"""Tests for the JobStore interface, exercised via the jobstore fixture."""

import pytest

from src.common.errors import MissingResourceError
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


def _create_queued(jobstore, qid: str = "iq__test", feature: str = "test_feature", deps: list = [], additional_info: dict = {}, resource: str | None = None):
    job = jobstore.create_job(CreateQueueItem(qid=qid, model=feature), auth="test-auth")
    jobstore.release_job(
        ReleaseJobRequest(id=job.id, params=_make_tag_args(feature), deps=deps, additional_info=additional_info, resource=resource or job.id),
        auth="test-auth",
    )
    return jobstore.get_job(job.id)


def _create_running(jobstore, **kwargs):
    job = _create_queued(jobstore, **kwargs)
    assert jobstore.claim_job(job.id, auth="test-auth")
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


def _list_all(jobstore) -> list:
    return jobstore.list_jobs(ListJobArgs(), auth="test-auth")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateAndList:
    def test_create_job_appears_in_list(self, jobstore):
        _create_queued(jobstore)
        jobs = _list_all(jobstore)
        assert len(jobs) == 1

    def test_created_job_has_correct_qid(self, jobstore):
        _create_queued(jobstore, qid="iq__abc")
        jobs = _list_all(jobstore)
        assert jobs[0].qid == "iq__abc"

    def test_created_job_has_correct_feature(self, jobstore):
        _create_queued(jobstore, feature="my_feature")
        jobs = _list_all(jobstore)
        assert jobs[0].params.feature == "my_feature"

    def test_created_job_initial_status_is_queued(self, jobstore):
        _create_queued(jobstore)
        # list_jobs returns QueueItem which doesn't carry status directly,
        # so verify via listing with status filter
        queued = jobstore.list_jobs(ListJobArgs(status="queued"), auth="test-auth")
        assert len(queued) == 1

    def test_multiple_jobs_all_listed(self, jobstore):
        _create_queued(jobstore, qid="iq__a")
        _create_queued(jobstore, qid="iq__b")
        jobs = _list_all(jobstore)
        assert len(jobs) == 2

    def test_additional_info_is_stored_and_retrieved(self, jobstore):
        info = {"key1": "value1", "key2": 42}
        _create_queued(jobstore, additional_info=info)
        jobs = _list_all(jobstore)
        assert len(jobs) == 1
        assert jobs[0].additional_info == info


class TestListFiltering:
    def test_filter_by_qid(self, jobstore):
        _create_queued(jobstore, qid="iq__alpha")
        _create_queued(jobstore, qid="iq__beta")

        results = jobstore.list_jobs(ListJobArgs(qid="iq__alpha"), auth="test-auth")
        assert len(results) == 1
        assert results[0].qid == "iq__alpha"

    def test_filter_by_status_returns_empty_when_no_match(self, jobstore):
        _create_queued(jobstore)
        results = jobstore.list_jobs(ListJobArgs(status="running"), auth="test-auth")
        assert results == []

    def test_filter_by_status_after_claim(self, jobstore):
        _create_queued(jobstore)
        job_id = _list_all(jobstore)[0].id
        jobstore.claim_job(job_id, auth="test-auth")

        running = jobstore.list_jobs(ListJobArgs(status="running"), auth="test-auth")
        assert len(running) == 1
        assert running[0].id == job_id


class TestClaimJob:
    def test_claim_queued_job_returns_true(self, jobstore):
        _create_queued(jobstore)
        job_id = _list_all(jobstore)[0].id
        assert jobstore.claim_job(job_id, auth="test-auth") is True

    def test_claim_moves_job_to_running(self, jobstore):
        _create_queued(jobstore)
        job_id = _list_all(jobstore)[0].id
        jobstore.claim_job(job_id, auth="test-auth")

        queued = jobstore.list_jobs(ListJobArgs(status="queued"), auth="test-auth")
        running = jobstore.list_jobs(ListJobArgs(status="running"), auth="test-auth")
        assert len(queued) == 0
        assert len(running) == 1

    def test_claim_already_running_job_returns_false(self, jobstore):
        _create_queued(jobstore)
        job_id = _list_all(jobstore)[0].id
        jobstore.claim_job(job_id, auth="test-auth")
        assert jobstore.claim_job(job_id, auth="test-auth") is False


class TestCompleteJob:
    def test_complete_running_job_as_succeeded(self, jobstore):
        job = _create_running(jobstore)

        assert jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth="test-auth")

        succeeded = jobstore.list_jobs(ListJobArgs(status="succeeded"), auth="test-auth")
        assert len(succeeded) == 1
        assert succeeded[0].id == job.id

    def test_complete_running_job_as_failed_with_error(self, jobstore):
        job = _create_running(jobstore)

        jobstore.complete_job(
            CompleteJobRequest(id=job.id, status="failed", error="something went wrong", status_details=_details()),
            auth="test-auth",
        )

        failed = jobstore.list_jobs(ListJobArgs(status="failed"), auth="test-auth")
        assert len(failed) == 1
        assert failed[0].error == "something went wrong"
        assert failed[0].status_details == _details()

    def test_running_job_cant_complete_as_cancelled(self, jobstore):
        job = _create_running(jobstore)
        assert not jobstore.complete_job(CompleteJobRequest(id=job.id, status="cancelled"), auth="test-auth")
        assert jobstore.get_job(job.id).status == "running"

    def test_queued_job_cant_complete(self, jobstore):
        job = _create_queued(jobstore)
        assert not jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth="test-auth")

    def test_ended_job_cant_complete(self, jobstore):
        job = _create_running(jobstore)
        jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth="test-auth")
        assert not jobstore.complete_job(CompleteJobRequest(id=job.id, status="failed"), auth="test-auth")
        assert jobstore.get_job(job.id).status == "succeeded"

    def test_failure_cascades_to_dependents(self, jobstore):
        parent = _create_running(jobstore)
        child = _create_queued(jobstore, deps=[parent.id])
        grandchild = _create_queued(jobstore, deps=[child.id])

        jobstore.complete_job(CompleteJobRequest(id=parent.id, status="failed"), auth="test-auth")

        assert jobstore.get_job(child.id).status == "failed"
        assert jobstore.get_job(grandchild.id).status == "failed"


class TestProgress:
    def test_update_progress_records_details(self, jobstore):
        job = _create_running(jobstore)
        job = jobstore.update_progress(job.id, _details(0.25), auth="test-auth")
        assert job.status == "running"
        assert job.status_details == _details(0.25)

    def test_update_progress_reports_stop_request(self, jobstore):
        job = _create_running(jobstore)
        jobstore.cancel_job(job.id, auth="test-auth")
        assert jobstore.update_progress(job.id, _details(), auth="test-auth").status == "cancelling"

    def test_update_progress_on_ended_job_returns_status(self, jobstore):
        job = _create_running(jobstore)
        jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth="test-auth")
        assert jobstore.update_progress(job.id, _details(), auth="test-auth").status == "succeeded"


class TestStopJob:
    def test_cancelling_job_completes_as_cancelled(self, jobstore):
        job = _create_running(jobstore)
        jobstore.cancel_job(job.id, auth="test-auth")
        assert jobstore.complete_job(CompleteJobRequest(id=job.id, status="cancelled"), auth="test-auth")
        assert jobstore.get_job(job.id).status == "cancelled"

    def test_cancelling_job_that_ended_on_its_own_completes_as_succeeded(self, jobstore):
        job = _create_running(jobstore)
        jobstore.cancel_job(job.id, auth="test-auth")
        assert jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth="test-auth")
        assert jobstore.get_job(job.id).status == "succeeded"

def test_dependency_listing(jobstore: JobStore):
    job = _create_queued(jobstore)
    child = _create_queued(jobstore, deps=[job.id])
    jobs = jobstore.list_jobs(ListJobArgs(), auth="test-auth")
    assert len(jobs) == 1
    assert jobs[0].id == job.id
    all_jobs = jobstore.list_jobs(ListJobArgs(include_unready=True), auth="test-auth")
    assert len(all_jobs) == 2


class TestPendingJobs:
    def test_pending_job_is_not_claimable(self, jobstore):
        job = jobstore.create_job(CreateQueueItem(qid="iq__test", model="m"), auth="test-auth")
        assert job.status == "pending"
        assert job.model == "m"
        assert job.params is None
        assert jobstore.list_jobs(ListJobArgs(status="queued"), auth="test-auth") == []
        assert jobstore.claim_job(job.id, auth="test-auth") is False

    def test_release_sets_params_and_queues(self, jobstore):
        dep = _create_queued(jobstore)
        job = jobstore.create_job(CreateQueueItem(qid="iq__test", model="m"), auth="test-auth")

        released = jobstore.release_job(
            ReleaseJobRequest(id=job.id, params=_make_tag_args("m"), deps=[dep.id], additional_info={"title": "t"}, resource="r"),
            auth="test-auth",
        )

        assert released is True
        job = jobstore.get_job(job.id)
        assert job.status == "queued"
        assert job.params == _make_tag_args("m")
        assert job.deps == [dep.id]
        assert job.additional_info == {"title": "t"}

    def test_release_fails_if_not_pending(self, jobstore):
        job = jobstore.create_job(CreateQueueItem(qid="iq__test", model="m"), auth="test-auth")
        jobstore.cancel_job(job.id, auth="test-auth")

        released = jobstore.release_job(
            ReleaseJobRequest(id=job.id, params=_make_tag_args("m"), deps=[], additional_info={}, resource="r"),
            auth="test-auth",
        )

        assert released is False
        assert jobstore.get_job(job.id).status == "cancelled"

    def test_pending_dependency_blocks_child(self, jobstore):
        parent = jobstore.create_job(CreateQueueItem(qid="iq__test", model="m"), auth="test-auth")
        _create_queued(jobstore, deps=[parent.id])
        assert jobstore.list_jobs(ListJobArgs(status="queued"), auth="test-auth") == []


class TestStopCancels:
    def test_stop_pending_job_cancels(self, jobstore):
        job = jobstore.create_job(CreateQueueItem(qid="iq__test", model="m"), auth="test-auth")
        jobstore.cancel_job(job.id, auth="test-auth", reason="duplicate")
        job = jobstore.get_job(job.id)
        assert job.status == "cancelled"
        assert job.error == "duplicate"

    def test_stop_queued_job_cancels(self, jobstore):
        job = _create_queued(jobstore)
        jobstore.cancel_job(job.id, auth="test-auth")
        assert jobstore.get_job(job.id).status == "cancelled"
        assert jobstore.claim_job(job.id, auth="test-auth") is False

    def test_stop_running_job_makes_it_cancelling(self, jobstore):
        job = _create_running(jobstore)
        jobstore.cancel_job(job.id, auth="test-auth")
        assert jobstore.get_job(job.id).status == "cancelling"

    def test_cancel_cascades_to_dependents(self, jobstore):
        parent = _create_running(jobstore)
        child = _create_queued(jobstore, deps=[parent.id])

        jobstore.cancel_job(parent.id, auth="test-auth")

        child = jobstore.get_job(child.id)
        assert child.status == "cancelled"
        assert parent.id in child.error


class TestDependencies:
    def test_child_is_claimable_once_parent_succeeds(self, jobstore):
        parent = _create_running(jobstore)
        child = _create_queued(jobstore, deps=[parent.id])
        assert jobstore.claim_job(child.id, auth="test-auth") is False

        jobstore.complete_job(CompleteJobRequest(id=parent.id, status="succeeded"), auth="test-auth")

        assert [j.id for j in jobstore.list_jobs(ListJobArgs(status="queued"), auth="test-auth")] == [child.id]
        assert jobstore.claim_job(child.id, auth="test-auth") is True


def test_list_limit(jobstore):
    for _ in range(3):
        _create_queued(jobstore)
    assert len(jobstore.list_jobs(ListJobArgs(status="queued", limit=2), auth="test-auth")) == 2


def test_delete_ended_job(jobstore):
    job = _create_running(jobstore)
    jobstore.complete_job(CompleteJobRequest(id=job.id, status="succeeded"), auth="test-auth")

    jobstore.delete_job(job.id, auth="test-auth")

    with pytest.raises(MissingResourceError):
        jobstore.get_job(job.id)
    assert _list_all(jobstore) == []


class TestResources:
    def test_release_refused_while_resource_active(self, jobstore):
        _create_queued(jobstore, resource="iq__test/m/video")
        job = jobstore.create_job(CreateQueueItem(qid="iq__test", model="m"), auth="test-auth")

        released = jobstore.release_job(
            ReleaseJobRequest(id=job.id, params=_make_tag_args("m"), deps=[], additional_info={}, resource="iq__test/m/video"),
            auth="test-auth",
        )

        assert released is False
        assert jobstore.get_job(job.id).status == "pending"

    def test_release_allowed_once_resource_ended(self, jobstore):
        first = _create_running(jobstore, resource="r")
        jobstore.complete_job(CompleteJobRequest(id=first.id, status="succeeded"), auth="test-auth")
        assert _create_queued(jobstore, resource="r").status == "queued"
