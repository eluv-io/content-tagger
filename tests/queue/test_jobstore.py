"""Tests for the JobStore interface, exercised via the jobstore fixture."""

from src.fetch.model import VideoScope
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.model import (
    CreateQueueItem,
    ListJobArgs,
    UpdateJobRequest,
)
from src.service.model import TagDetails


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
        max_fetch_retries=3,
    )


def _make_create_item(qid: str = "iq__test", feature: str = "test_feature", additional_info: dict = {}) -> CreateQueueItem:
    return CreateQueueItem(
        qid=qid,
        params=_make_tag_args(feature),
        status_details=None,
        additional_info=additional_info,
    )


def _list_all(jobstore) -> list:
    return jobstore.list_jobs(ListJobArgs(), auth="test-auth")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateAndList:
    def test_create_job_appears_in_list(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        jobs = _list_all(jobstore)
        assert len(jobs) == 1

    def test_created_job_has_correct_qid(self, jobstore):
        jobstore.create_job(_make_create_item(qid="iq__abc"), auth="test-auth")
        jobs = _list_all(jobstore)
        assert jobs[0].qid == "iq__abc"

    def test_created_job_has_correct_feature(self, jobstore):
        jobstore.create_job(_make_create_item(feature="my_feature"), auth="test-auth")
        jobs = _list_all(jobstore)
        assert jobs[0].params.feature == "my_feature"

    def test_created_job_initial_status_is_queued(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        # list_jobs returns QueueItem which doesn't carry status directly,
        # so verify via listing with status filter
        queued = jobstore.list_jobs(ListJobArgs(status="queued"), auth="test-auth")
        assert len(queued) == 1

    def test_multiple_jobs_all_listed(self, jobstore):
        jobstore.create_job(_make_create_item(qid="iq__a"), auth="test-auth")
        jobstore.create_job(_make_create_item(qid="iq__b"), auth="test-auth")
        jobs = _list_all(jobstore)
        assert len(jobs) == 2

    def test_additional_info_is_stored_and_retrieved(self, jobstore):
        info = {"key1": "value1", "key2": 42}
        jobstore.create_job(_make_create_item(additional_info=info), auth="test-auth")
        jobs = _list_all(jobstore)
        assert len(jobs) == 1
        assert jobs[0].additional_info == info


class TestListFiltering:
    def test_filter_by_qid(self, jobstore):
        jobstore.create_job(_make_create_item(qid="iq__alpha"), auth="test-auth")
        jobstore.create_job(_make_create_item(qid="iq__beta"), auth="test-auth")

        results = jobstore.list_jobs(ListJobArgs(qid="iq__alpha"), auth="test-auth")
        assert len(results) == 1
        assert results[0].qid == "iq__alpha"

    def test_filter_by_status_returns_empty_when_no_match(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        results = jobstore.list_jobs(ListJobArgs(status="running"), auth="test-auth")
        assert results == []

    def test_filter_by_status_after_claim(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        job_id = _list_all(jobstore)[0].id
        jobstore.claim_job(job_id, auth="test-auth")

        running = jobstore.list_jobs(ListJobArgs(status="running"), auth="test-auth")
        assert len(running) == 1
        assert running[0].id == job_id


class TestClaimJob:
    def test_claim_queued_job_returns_true(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        job_id = _list_all(jobstore)[0].id
        assert jobstore.claim_job(job_id, auth="test-auth") is True

    def test_claim_moves_job_to_running(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        job_id = _list_all(jobstore)[0].id
        jobstore.claim_job(job_id, auth="test-auth")

        queued = jobstore.list_jobs(ListJobArgs(status="queued"), auth="test-auth")
        running = jobstore.list_jobs(ListJobArgs(status="running"), auth="test-auth")
        assert len(queued) == 0
        assert len(running) == 1

    def test_claim_already_running_job_returns_false(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        job_id = _list_all(jobstore)[0].id
        jobstore.claim_job(job_id, auth="test-auth")
        assert jobstore.claim_job(job_id, auth="test-auth") is False


class TestUpdateJob:
    def test_update_status_to_succeeded(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        job_id = _list_all(jobstore)[0].id
        jobstore.claim_job(job_id, auth="test-auth")

        jobstore.update_job(
            UpdateJobRequest(
                id=job_id,
                status="succeeded",
                status_details=None,
            ),
            auth="test-auth",
        )

        succeeded = jobstore.list_jobs(ListJobArgs(status="succeeded"), auth="test-auth")
        assert len(succeeded) == 1
        assert succeeded[0].id == job_id

    def test_update_status_to_failed_with_error(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        job_id = _list_all(jobstore)[0].id

        jobstore.update_job(
            UpdateJobRequest(
                id=job_id,
                status="failed",
                error="something went wrong",
                status_details=None,
            ),
            auth="test-auth",
        )

        failed = jobstore.list_jobs(ListJobArgs(status="failed"), auth="test-auth")
        assert len(failed) == 1
        assert failed[0].error == "something went wrong"


class TestStopJob:
    def test_stop_sets_stop_requested(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        job_id = _list_all(jobstore)[0].id

        jobstore.stop_job(job_id, auth="test-auth")

        jobs = _list_all(jobstore)
        assert jobs[0].stop_requested is True

    def test_stop_requested_is_false_before_stop(self, jobstore):
        jobstore.create_job(_make_create_item(), auth="test-auth")
        jobs = _list_all(jobstore)
        assert jobs[0].stop_requested is False
