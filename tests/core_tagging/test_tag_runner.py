"""Tests for TagRunner + QueueClient.

Mirrors the test_fabric_tagger.py tests but work is submitted through the
QueueClient and processed asynchronously by the TagRunner.
"""

import time
import pytest

from src.common.errors import MissingResourceError
from src.tagging.fabric_tagging.model import TagJobStatusReport

def _wait_for_status(
    client,
    qhit: str,
    target_status: str,
    timeout: float = 10.0,
    interval: float = 0.15,
) -> list[TagJobStatusReport]:
    """Poll until every report for *qhit* reaches *target_status* or timeout."""
    deadline = time.time() + timeout
    reports: list[TagJobStatusReport] = []
    while time.time() < deadline:
        try:
            reports = client.status(qhit)
        except Exception:
            time.sleep(interval)
            continue
        if reports and all(r.status == target_status for r in reports):
            return reports
        time.sleep(interval)
    return reports


def _status_for(
    reports: list[TagJobStatusReport],
    model: str,
    stream: str | None = None,
) -> TagJobStatusReport:
    matches = [r for r in reports if r.model == model and (stream is None or r.stream == stream)]
    assert matches, f"Missing status for model={model}, stream={stream}"
    return matches[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQueueTag:
    def test_tag_returns_started(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        result = queue_client.tag(q, args)
        assert result.started is True
        assert result.message == "Job enqueued"

    def test_job_completes(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        queue_client.tag(q, args)

        reports = _wait_for_status(queue_client, q.qhit, "Completed")
        assert len(reports) >= 1
        assert any(r.status == "Completed" for r in reports)

    def test_multiple_jobs_complete(self, queue_client, q, sample_tag_args, tag_runner):
        for args in sample_tag_args:
            queue_client.tag(q, args)

        reports = _wait_for_status(queue_client, q.qhit, "Completed")
        completed = [r for r in reports if r.status == "Completed"]
        assert len(completed) == len(sample_tag_args)


class TestQueueStatus:
    def test_status_after_enqueue(self, queue_client, q, make_tag_args, tag_runner):
        """Before the runner picks it up we get a synthesised status."""
        args = make_tag_args(feature="caption", stream="video")
        queue_client.tag(q, args)

        reports = queue_client.status(q.qhit)
        assert len(reports) == 1
        assert reports[0].model == "caption"

    def test_status_no_jobs(self, queue_client, tag_runner):
        with pytest.raises(MissingResourceError):
            queue_client.status("iq__nonexistent")

    def test_status_with_completed_jobs(self, queue_client, q, sample_tag_args, tag_runner):
        for args in sample_tag_args:
            queue_client.tag(q, args)

        reports = _wait_for_status(queue_client, q.qhit, "Completed")
        assert len(reports) == 2
        assert any(r.model == "caption" for r in reports)
        assert any(r.model == "asr" for r in reports)


class TestQueueStop:
    def test_stop_returns_result(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        queue_client.tag(q, args)

        results = queue_client.stop(q.qhit, "caption", None)
        assert len(results) == 1
        assert results[0].message == "Stop requested"

    def test_stop_wrong_feature_raises_exception(self, queue_client, q, make_tag_args, tag_runner):
        args = make_tag_args(feature="caption", stream="video")
        queue_client.tag(q, args)

        with pytest.raises(MissingResourceError):
            queue_client.stop(q.qhit, "nonexistent", None)
