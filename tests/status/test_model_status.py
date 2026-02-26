import pytest
from unittest.mock import Mock

from src.status.model_status import get_model_status
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Batch
from src.tags.track_resolver import TrackResolver, TrackResolverConfig, TrackArgs
from src.common.errors import BadRequestError, MissingResourceError


def make_batch(
    id: str,
    track: str,
    timestamp: float,
    all_sources: list,
    downloaded_sources: list,
    tagged_sources: list,
    source_qid: str = "iq__src",
    job_status: str = "Completed",
) -> Batch:
    return Batch(
        id=id,
        qhit="iq__test",
        track=track,
        timestamp=timestamp,
        author="tagger",
        additional_info={
            "tagger": {
                "source_qid": source_qid,
                "params": {"feature": "llava", "run_config": {}, "scope": "full", "replace": False, "destination_qid": "iq__dest", "max_fetch_retries": 3},
                "job_status": {"status": job_status, "time_ran": "1h 0m 0s"},
                "upload_status": {
                    "all_sources": all_sources,
                    "downloaded_sources": downloaded_sources,
                    "tagged_sources": tagged_sources,
                },
            }
        },
    )

def test_single_batch_full_completion(track_resolver, mock_tagstore):
    sources = ["s1", "s2", "s3"]
    batches = [make_batch("b1", "llava_track", 1000.0, sources, sources, sources)]
    tagstore = mock_tagstore(batches)

    result = get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)

    assert result.summary.model == "llava"
    assert result.summary.track == "llava_track"
    assert result.summary.tagging_progress == pytest.approx(1.0)
    assert result.summary.num_content_parts == 3


def test_single_batch_partial_completion(track_resolver, mock_tagstore):
    batches = [make_batch("b1", "llava_track", 1000.0, ["s1", "s2", "s3", "s4"], ["s1", "s2"], ["s1"])]
    tagstore = mock_tagstore(batches)

    result = get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)

    assert result.summary.tagging_progress == pytest.approx(0.25)
    assert result.summary.num_content_parts == 4


def test_multiple_batches_union_sources(track_resolver, mock_tagstore):
    """tagging_progress and num_content_parts are computed across all batches."""
    batches = [
        make_batch("b1", "llava_track", 1000.0, ["s1", "s2", "s3", "s4"], ["s1", "s2"], ["s1"]),
        make_batch("b2", "llava_track", 2000.0, ["s1", "s2", "s3"], ["s3"], ["s3", "s4"]),
    ]
    tagstore = mock_tagstore(batches)

    result = get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)

    assert result.summary.num_content_parts == 4
    assert result.summary.tagging_progress == pytest.approx(0.75)
    assert len(result.jobs) == 2


def test_num_content_parts_is_max_not_union(track_resolver, mock_tagstore):
    """num_content_parts takes the max batch size, not the union count."""
    batches = [
        make_batch("b1", "llava_track", 1000.0, ["s1", "s2", "s3"], [], []),
        make_batch("b2", "llava_track", 2000.0, ["s1", "s2"], [], []),
    ]
    tagstore = mock_tagstore(batches)

    result = get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)

    assert result.summary.num_content_parts == 3


def test_job_detail_fields(track_resolver, mock_tagstore):
    batches = [make_batch("b1", "llava_track", 1000.0, ["s1", "s2"], ["s1", "s2"], ["s1"], source_qid="iq__src1", job_status="Completed")]
    tagstore = mock_tagstore(batches)

    result = get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)

    assert len(result.jobs) == 1
    job = result.jobs[0]
    assert job.source_qid == "iq__src1"
    assert job.job_status.status == "Completed"
    assert job.upload_status is not None
    assert job.upload_status.num_job_parts == 2
    assert job.upload_status.num_tagged_parts == 1


def test_job_upload_status_shows_counts_not_lists(track_resolver, mock_tagstore):
    batches = [make_batch("b1", "llava_track", 1000.0, ["s1", "s2", "s3"], ["s1", "s2"], ["s1"])]
    tagstore = mock_tagstore(batches)

    result = get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)

    upload = result.jobs[0].upload_status
    assert upload is not None
    assert isinstance(upload.num_job_parts, int)
    assert isinstance(upload.num_tagged_parts, int)
    assert upload.num_job_parts == 2
    assert upload.num_tagged_parts == 1


def test_no_batches_raises_missing_resource(track_resolver, mock_tagstore):
    tagstore = mock_tagstore([])

    with pytest.raises(MissingResourceError):
        get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)


def test_only_other_track_batches_raises_missing_resource(track_resolver, mock_tagstore):
    batches = [make_batch("b1", "whisper_track", 1000.0, ["s1"], ["s1"], ["s1"])]
    tagstore = mock_tagstore(batches)

    with pytest.raises(MissingResourceError):
        get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)


def test_no_upload_status_in_batch(track_resolver, mock_tagstore):
    """Batches without upload_status should still appear as jobs with zero counts."""
    batch = Batch(
        id="b1",
        qhit="iq__test",
        track="llava_track",
        timestamp=1000.0,
        author="tagger",
        additional_info={
            "tagger": {
                "source_qid": "iq__src",
                "params": {"feature": "llava", "run_config": {}, "scope": "full", "replace": False, "destination_qid": "iq__dest", "max_fetch_retries": 3},
                "job_status": {"status": "Failed"},
            }
        },
    )
    tagstore = mock_tagstore([batch])

    with pytest.raises(BadRequestError):
        get_model_status(Mock(qhit="iq__test"), "llava", tagstore, track_resolver)