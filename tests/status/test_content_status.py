import pytest
from unittest.mock import Mock

from src.status.content_status import get_content_summary
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Batch
from src.tags.track_resolver import TrackResolver, TrackResolverConfig, TrackArgs


def make_batch(id: str, track: str, timestamp: float, all_sources: list, tagged_sources: list) -> Batch:
    return Batch(
        id=id,
        qhit="iq__test",
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


@pytest.fixture
def track_resolver() -> TrackResolver:
    return TrackResolver(
        TrackResolverConfig(
            mapping={
                "llava": TrackArgs(name="llava_track", label="LLaVA"),
                "whisper": TrackArgs(name="whisper_track", label="Whisper"),
            }
        )
    )


def _mock_tagstore(batches: list[Batch]) -> Tagstore:
    tagstore = Mock(spec=Tagstore)
    tagstore.find_batches.return_value = [b.id for b in batches]
    tagstore.get_batch.side_effect = {b.id: b for b in batches}.get
    return tagstore


def test_single_batch_full_completion(track_resolver):
    sources = ["s1", "s2", "s3"]
    batches = [make_batch("b1", "llava_track", 1000.0, sources, sources)]
    tagstore = _mock_tagstore(batches)

    result = get_content_summary("iq__test", tagstore, track_resolver)

    assert len(result.models) == 1
    m = result.models[0]
    assert m.model == "llava"
    assert m.track == "llava_track"
    assert m.last_run == 1000.0
    assert m.percent_completion == 1.0


def test_single_batch_partial_completion(track_resolver):
    batches = [make_batch("b1", "llava_track", 1000.0, ["s1", "s2", "s3", "s4"], ["s1", "s2"])]
    tagstore = _mock_tagstore(batches)

    result = get_content_summary("iq__test", tagstore, track_resolver)

    assert result.models[0].percent_completion == pytest.approx(0.5)


def test_multiple_batches_union_sources(track_resolver):
    """Completion is computed by unioning sources across all batches for a track."""
    batches = [
        make_batch("b1", "llava_track", 1000.0, ["s1", "s2"], ["s1"]),
        make_batch("b2", "llava_track", 2000.0, ["s3", "s4"], ["s3", "s4"]),
    ]
    tagstore = _mock_tagstore(batches)

    result = get_content_summary("iq__test", tagstore, track_resolver)

    assert len(result.models) == 1
    m = result.models[0]
    # 3 tagged out of 4 total across both batches
    assert m.percent_completion == pytest.approx(0.75)
    assert m.last_run == 2000.0  # latest batch timestamp


def test_multiple_tracks(track_resolver):
    batches = [
        make_batch("b1", "llava_track", 1000.0, ["s1", "s2"], ["s1", "s2"]),
        make_batch("b2", "whisper_track", 500.0, ["s1", "s2"], ["s1"]),
    ]
    tagstore = _mock_tagstore(batches)

    result = get_content_summary("iq__test", tagstore, track_resolver)

    by_model = {m.model: m for m in result.models}
    assert set(by_model.keys()) == {"llava", "whisper"}
    assert by_model["llava"].percent_completion == 1.0
    assert by_model["whisper"].percent_completion == pytest.approx(0.5)


def test_no_upload_status(track_resolver):
    """Batches without upload_status should yield 0% completion."""
    batch = Batch(
        id="b1",
        qhit="iq__test",
        track="llava_track",
        timestamp=1000.0,
        author="tagger",
        additional_info={"tagger": {}},
    )
    tagstore = _mock_tagstore([batch])

    result = get_content_summary("iq__test", tagstore, track_resolver)

    assert result.models[0].percent_completion == 0.0


def test_unmapped_track_falls_back_to_track_name(track_resolver):
    """A track with no mapping in the resolver should use the track name as the model name."""
    batches = [make_batch("b1", "unknown_model", 1000.0, ["s1"], ["s1"])]
    tagstore = _mock_tagstore(batches)

    result = get_content_summary("iq__test", tagstore, track_resolver)

    assert result.models[0].model == "unknown_model"


def test_no_batches_returns_empty(track_resolver):
    tagstore = _mock_tagstore([])

    result = get_content_summary("iq__test", tagstore, track_resolver)

    assert result.models == []
