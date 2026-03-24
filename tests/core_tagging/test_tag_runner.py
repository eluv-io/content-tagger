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
from src.tagging.fabric_tagging.queue.model import ListJobArgs

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


def _status_for(
    reports: list[TagJobStatusResult],
    model: str,
    stream: str | None = None,
) -> TagJobStatusResult:
    matches = [r for r in reports if r.model == model]
    assert matches, f"Missing status for model={model}, stream={stream}"
    return matches[0]

class TestQueueTag:
    def test_tag_returns_started(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        result = queue_client.tag(q, args)
        assert result.started is True
        assert result.message == "Job enqueued"

    def test_job_completes(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        queue_client.tag(q, args)

        reports = _wait_for_status(queue_client, q.qid, "succeeded")
        assert len(reports) >= 1
        assert any(r.status == "succeeded" for r in reports)

    def test_multiple_jobs_complete(self, queue_client, q, sample_tag_args, tag_runner):
        for args in sample_tag_args:
            queue_client.tag(q, args)

        reports = _wait_for_status(queue_client, q.qid, "succeeded")
        completed = [r for r in reports if r.status == "succeeded"]
        assert len(completed) == len(sample_tag_args)


class TestQueueStatus:
    def test_status_after_enqueue(self, queue_client, q, make_tag_args, make_status_args, tag_runner):
        """Before the runner picks it up we get a synthesised status."""
        args = make_tag_args(feature="caption", stream="video")
        queue_client.tag(q, args)

        reports = queue_client.status(make_status_args(qid=q.qid))
        assert len(reports) == 1
        assert reports[0].model == "caption"

    def test_status_no_jobs(self, queue_client, make_status_args, tag_runner):
        with pytest.raises(MissingResourceError):
            queue_client.status(make_status_args(qid="iq__nonexistent"))

    def test_status_with_completed_jobs(self, queue_client, q, sample_tag_args, tag_runner):
        for args in sample_tag_args:
            queue_client.tag(q, args)

        reports = _wait_for_status(queue_client, q.qid, "succeeded")
        assert len(reports) == 2
        assert any(r.model == "caption" for r in reports)
        assert any(r.model == "asr" for r in reports)


class TestQueueStop:
    def test_stop_marks_cancelled(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        queue_client.tag(q, args)

        results = queue_client.stop(q.qid, "caption")
        assert len(results) == 1
        assert results[0].message == "Stop requested"
        time.sleep(0.25)
        # check that job is marked cancelled in jobstore
        jobstore = tag_runner.jobstore
        assert jobstore.list_jobs(ListJobArgs(status="cancelled"), auth="")

    def test_stop_wrong_feature_raises_exception(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        queue_client.tag(q, args)

        with pytest.raises(MissingResourceError):
            queue_client.stop(q.qid, "nonexistent")


def test_stop_runner(queue_client, q, make_tag_args, tag_runner):
    args = make_tag_args(feature="caption", stream="video")
    queue_client.tag(q, args)
    time.sleep(0.25)
    tag_runner.stop()
    # check that job is marked cancelled in jobstore
    jobstore = tag_runner.jobstore
    assert jobstore.list_jobs(ListJobArgs(status="cancelled"), auth="")

def test_stop_running_job(queue_client, q, make_tag_args, tag_runner):
    args = make_tag_args(feature="caption", stream="video")
    queue_client.tag(q, args)
    time.sleep(0.25)
    queue_client.stop(q.qid, "caption")
    time.sleep(0.25)
    # check that job is marked cancelled in jobstore
    jobstore = tag_runner.jobstore
    assert jobstore.list_jobs(ListJobArgs(status="cancelled"), auth="")

def test_worker_tag_fails(queue_client, q, make_tag_args, tag_runner):
    tag_runner.tagger.tag = Mock(side_effect=Exception("Tagging failed"))

    args = make_tag_args(feature="caption", stream="video")
    queue_client.tag(q, args)
    time.sleep(0.25)

    # check that job is marked failed in jobstore
    jobstore = tag_runner.jobstore
    failed_jobs = jobstore.list_jobs(ListJobArgs(status="failed"), auth="")
    assert len(failed_jobs) == 1
    assert failed_jobs[0].error == "Tagging failed"