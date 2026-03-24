import pytest
from unittest.mock import Mock

from src.tags.tagstore.model import Batch


def make_batch(id: str, track: str, timestamp: float, all_sources: list, tagged_sources: list) -> Batch:
    return Batch(
        id=id,
        qid="iq__test",
        track=track,
        timestamp=timestamp,
        author="tagger",
        additional_info={
            "tagger": {
                "upload_status": {
                    "all_sources": all_sources,
                    "downloaded_sources": all_sources,
                    "tagged_sources": tagged_sources,
                }
            }
        },
    )


def test_single_batch_full_completion(get_status_service):
    sources = ["s1", "s2", "s3"]
    batches = [make_batch("b1", "llava_track", 1000.0, sources, sources)]
    service = get_status_service(batches)

    result = service.get_content_summary(Mock(qid="iq__test"))

    assert len(result.models) == 1
    m = result.models[0]
    assert m.model == "llava"
    assert m.track == "llava_track"
    assert m.last_run.startswith('1970')
    assert m.percent_completion == 1.0


def test_single_batch_partial_completion(get_status_service):
    batches = [make_batch("b1", "llava_track", 1000.0, ["s1", "s2", "s3", "s4"], ["s1", "s2"])]
    service = get_status_service(batches)

    result = service.get_content_summary(Mock(qid="iq__test"))

    assert result.models[0].percent_completion == pytest.approx(0.5)


def test_multiple_batches_union_sources(get_status_service):
    """Completion is computed by unioning sources across all batches for a track."""
    batches = [
        make_batch("b1", "llava_track", 1000.0, ["s1", "s2"], ["s1"]),
        make_batch("b2", "llava_track", 2000.0, ["s3", "s4"], ["s3", "s4"]),
    ]
    service = get_status_service(batches)

    result = service.get_content_summary(Mock(qid="iq__test"))

    assert len(result.models) == 1
    m = result.models[0]
    # 3 tagged out of 4 total across both batches
    assert m.percent_completion == pytest.approx(0.75)
    assert m.last_run.startswith('1970')


def test_multiple_tracks(get_status_service):
    batches = [
        make_batch("b1", "llava_track", 1000.0, ["s1", "s2"], ["s1", "s2"]),
        make_batch("b2", "whisper_track", 500.0, ["s1", "s2"], ["s1"]),
    ]
    service = get_status_service(batches)

    result = service.get_content_summary(Mock(qid="iq__test"))

    by_model = {m.model: m for m in result.models}
    assert set(by_model.keys()) == {"llava", "whisper"}
    assert by_model["llava"].percent_completion == 1.0
    assert by_model["whisper"].percent_completion == pytest.approx(0.5)


def test_no_upload_status(get_status_service):
    """Batches without upload_status should yield 0% completion."""
    batch = Batch(
        id="b1",
        qid="iq__test",
        track="llava_track",
        timestamp=1000.0,
        author="tagger",
        additional_info={"tagger": {}},
    )
    service = get_status_service([batch])

    result = service.get_content_summary(Mock(qid="iq__test"))

    assert result.models[0].percent_completion == 0.0


def test_unmapped_track_falls_back_to_track_name(get_status_service):
    """A track with no mapping in the resolver should use the track name as the model name."""
    batches = [make_batch("b1", "unknown_model", 1000.0, ["s1"], ["s1"])]
    service = get_status_service(batches)

    result = service.get_content_summary(Mock(qid="iq__test"))

    assert result.models[0].model == "unknown_model"


def test_no_batches_returns_empty(get_status_service):
    service = get_status_service([])

    result = service.get_content_summary(Mock(qid="iq__test"))

    assert result.models == []
