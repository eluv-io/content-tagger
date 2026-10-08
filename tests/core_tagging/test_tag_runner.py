"""Tests for TagRunner + QueueService.

Mirrors the test_fabric_tagger.py tests but work is submitted through the
QueueService and processed asynchronously by the TagRunner.
"""

import time
from unittest.mock import Mock
import pytest

from src.common.errors import MissingResourceError
from src.service.impl.queue_based import QueueService
from src.service.model import *
from src.tag_containers.containers import TagContainer
from src.tagging.fabric_tagging.queue.model import ListJobArgs
from src.common.content import Content
from tests.core_tagging.conftest import FakeTagContainer, enqueue

def _wait_for_status(
    client: QueueService,
    qid: str,
    target_status: str,
    timeout: float = 10.0,
    interval: float = 0.15,
) -> list[TagJobStatusResult]:
    """Poll until every report for *qid* reaches *target_status* or timeout."""
    deadline = time.time() + timeout
    req = StatusArgs(
        qid=qid,
        user=None,
        tenant=None,
        title=None
    )
    reports = []
    while time.time() < deadline:
        try:
            reports = client.status(req)
        except Exception:
            time.sleep(interval)
            continue
        if reports and all(r.status == target_status for r in reports):
            break
        time.sleep(interval)
    return reports

class TestQueueTag:
    def test_tag_returns_started(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        job_id = enqueue(queue_client, q, [args])[0]
        assert queue_client.jobstore.get_job(job_id).status in ("queued", "running", "succeeded")

    def test_job_completes(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        enqueue(queue_client, q, [args])
        
        reports = _wait_for_status(queue_client, q.qid, "succeeded")
        assert len(reports) >= 1
        assert any(r.status == "succeeded" for r in reports)

    def test_multiple_jobs_complete(self, queue_client, q, sample_tag_args, tag_runner):
        enqueue(queue_client, q, sample_tag_args)

        reports = _wait_for_status(queue_client, q.qid, "succeeded")
        completed = [r for r in reports if r.status == "succeeded"]
        assert len(completed) == len(sample_tag_args)


class TestQueueStatus:
    def test_status_after_enqueue(self, queue_client, q, make_tag_args, make_status_args, tag_runner):
        """Before the runner picks it up we get a synthesised status."""
        args = make_tag_args(feature="caption", stream="video")
        enqueue(queue_client, q, [args])

        reports = queue_client.status(make_status_args(qid=q.qid))
        assert len(reports) == 1
        assert reports[0].model == "caption"

    def test_status_no_jobs(self, queue_client, make_status_args, tag_runner):
        with pytest.raises(MissingResourceError):
            queue_client.status(make_status_args(qid="iq__nonexistent"))

    def test_status_with_completed_jobs(self, queue_client, q, sample_tag_args, tag_runner):
        for args in sample_tag_args:
            enqueue(queue_client, q, [args])

        reports = _wait_for_status(queue_client, q.qid, "succeeded")
        assert len(reports) == 2
        assert any(r.model == "caption" for r in reports)
        assert any(r.model == "asr" for r in reports)


class TestQueueStop:
    def test_stop_marks_cancelled(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        enqueue(queue_client, q, [args])

        results = queue_client.stop(q.qid, "caption")
        assert len(results) == 1
        assert results[0].message == "Stop requested"
        time.sleep(0.25)
        # check that job is marked cancelled in jobstore
        jobstore = tag_runner.jobstore
        assert jobstore.list_jobs(ListJobArgs(qid=q.qid, status="cancelled"), auth="")

    def test_stop_wrong_feature_raises_exception(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        enqueue(queue_client, q, [args])

        with pytest.raises(MissingResourceError):
            queue_client.stop(q.qid, "nonexistent")


def test_stop_runner(queue_client, q, make_tag_args, tag_runner):
    args = make_tag_args(feature="caption", stream="video")
    enqueue(queue_client, q, [args])
    time.sleep(0.25)
    tag_runner.stop()
    # a running job can't be cancelled by its worker, so a shut down fails it
    jobstore = tag_runner.jobstore
    failed = jobstore.list_jobs(ListJobArgs(qid=q.qid, status="failed"), auth="")
    assert failed
    assert failed[0].error == "tagger worker service was shut down or restarted"

def test_stop_running_job(queue_client, q, make_tag_args, tag_runner):
    args = make_tag_args(feature="caption", stream="video")
    enqueue(queue_client, q, [args])
    time.sleep(0.25)
    queue_client.stop(q.qid, "caption")
    time.sleep(0.5)
    # check that job is marked cancelled in jobstore
    jobstore = tag_runner.jobstore
    assert jobstore.list_jobs(ListJobArgs(qid=q.qid, status="cancelled"), auth="")
    
def test_job_progress(queue_client, q, make_tag_args, tag_runner):
    args = make_tag_args(feature="caption", stream="video")
    #def set_report_progress(container: FakeTagContainer) -> FakeTagContainer:
    #    container.report_progress = True
    #    return container
    #tag_runner.tagger.cregistry.get = 
    enqueue(queue_client, q, [args])
    time.sleep(2)
    status = queue_client.status(StatusArgs(q.qid, None, None, None))[0]
    assert status.tagger_details.progress == 1.0

def test_worker_tag_fails(queue_client, q, make_tag_args, tag_runner):
    tag_runner.tagger.tag = Mock(side_effect=Exception("Tagging failed"))

    args = make_tag_args(feature="caption", stream="video")
    enqueue(queue_client, q, [args])
    time.sleep(0.25)

    # check that job is marked failed in jobstore
    jobstore = tag_runner.jobstore
    failed_jobs = jobstore.list_jobs(ListJobArgs(qid=q.qid, status="failed"), auth="")
    assert len(failed_jobs) == 1
    assert failed_jobs[0].error == "Tagging failed"


def test_max_jobs_limits_concurrency(queue_client, q, make_tag_args, tag_runner):
    """TagRunner should not claim more than max_jobs (=2) jobs concurrently."""

    # Enqueue 3 jobs for distinct models (jobs take ~0.35s to complete)
    for feature in ("caption", "asr", "ocr"):
        enqueue(queue_client, q, [make_tag_args(feature=feature)])

    # Wait for the runner to poll once but not long enough for jobs to finish
    time.sleep(0.15)

    jobstore = tag_runner.jobstore
    running = jobstore.list_jobs(ListJobArgs(qid=q.qid, status="running"), auth="")
    queued = jobstore.list_jobs(ListJobArgs(qid=q.qid, status="queued"), auth="")

    assert len(running) <= 2, f"Expected at most 2 running jobs (max_jobs=2), got {len(running)}"
    assert len(queued) >= 1, f"Expected at least 1 job still queued, got {len(queued)}"
