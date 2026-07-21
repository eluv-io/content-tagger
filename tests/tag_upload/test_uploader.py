

from dataclasses import asdict, dataclass
from unittest.mock import Mock

import pytest

from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs, TagContentStatusReport
from src.tagging.uploading.uploader import UploadSession
from src.tags.track_resolver import TrackArgs, TrackResolver, LabelResolverConfig

def test_upload_tags(upload_session, get_tag):
    tags = [
        get_tag(model_track="", text="hello world"),
        get_tag(model_track="pretty", text="test tag", start_time=100, end_time=200)
    ]

    upload_session.upload_tags(tags=tags, tagged_sources=[t.source_media for t in tags])

    ts = upload_session.tagstore
    track = ts.get_track(name="speech_to_text", q=upload_session.dest_q)

    # check that tracks exist
    assert track is not None
    assert track.name == "speech_to_text"
    assert track.label == "Speech to Text"

    track = ts.get_track(name="pretty", q=upload_session.dest_q)
    assert track is not None
    assert track.name == "pretty"
    assert track.label == "Pretty Speech"

    ts_tags = ts.find_tags(q=upload_session.dest_q)
    assert len(ts_tags) == 2

    speech_tag = ts.find_tags(q=upload_session.dest_q, track="speech_to_text")[0]
    assert speech_tag.text == "hello world"

    pretty_tag = ts.find_tags(q=upload_session.dest_q, track="pretty")[0]
    assert pretty_tag.text == "test tag"
    assert pretty_tag.start_time == 100
    assert pretty_tag.end_time == 200

    # make sure the tags are in the same batch
    batch = ts_tags[0].batch_id
    assert all(t.batch_id == batch for t in ts_tags)

def test_upload_report(upload_session, get_tag):
    tags = [
        get_tag(model_track="asr", text="hello world"),
    ]

    upload_session.upload_tags(tags=tags, tagged_sources=[t.source_media for t in tags])

    @dataclass
    class MockParams:
        feature: str

    @dataclass
    class MockReportParams:
        params: MockParams

    report = MockReportParams(params=MockParams(feature="asr"))

    upload_session.upload_report(report=report)

    ts = upload_session.tagstore
    batch = ts.find_batches(q=upload_session.dest_q, model="asr")[0]

    assert batch

    db_batch = ts.get_batch(batch_id=batch, q=upload_session.dest_q)
    assert db_batch is not None
    assert db_batch.additional_info == {"tagger": asdict(report)}
    assert db_batch.model == "asr"
    assert db_batch.additional_info["tagger"]
    assert "params" in db_batch.additional_info["tagger"]

def test_get_uploaded_sources(upload_session, get_tag):
    tags = [
        get_tag(model_track="asr", text="hello world", source_media="/path/to/source1.mp4"),
        get_tag(model_track="caption", text="test tag", start_time=100, end_time=200, source_media="/path/to/source2.mp4")
    ]

    tagged_sources = ["source1", "source2"]

    upload_session.upload_tags(tags=tags, tagged_sources=tagged_sources)

    uploaded_sources = upload_session.get_uploaded_sources()

    assert set(uploaded_sources) == {"source1", "source2"}

def test_get_uploaded_sources_upload_fails(upload_session, get_tag):
    tags = [
        get_tag(model_track="asr", text="hello world", source_media="/path/to/source1.mp4"),
    ]

    tagged_sources = ["source1"]

    upload_session.tagstore.create_track = Mock(side_effect=Exception("upload failed"))

    with pytest.raises(Exception):
        upload_session.upload_tags(tags=tags, tagged_sources=tagged_sources)
    
    uploaded_sources = upload_session.get_uploaded_sources()
    assert not uploaded_sources

def test_upload_empty_tags_get_sources(upload_session):
    tagged_sources = ["source1", "source2"]

    upload_session.upload_tags(tags=[], tagged_sources=tagged_sources)

    uploaded_sources = upload_session.get_uploaded_sources()

    assert set(uploaded_sources) == {"source1", "source2"}

def test_reupload_replaces_tags_for_source(upload_session, track_resolver, get_tag):
    ts = upload_session.tagstore
    q = upload_session.dest_q

    # first tagging run processes both sources and writes tags for them
    old_tags = [
        get_tag(model_track="asr", text="old tag 1", source_media="source1"),
        get_tag(model_track="asr", text="old tag 2", source_media="source1"),
        get_tag(model_track="asr", text="old tag 3", source_media="source2"),
    ]
    upload_session.upload_tags(tags=old_tags, tagged_sources=["source1", "source2"])

    assert {t.text for t in ts.find_tags(q=q)} == {"old tag 1", "old tag 2", "old tag 3"}

    # a fresh run reprocesses only source1 and emits different tags for it
    second_session = UploadSession(
        feature="asr",
        track_resolver=track_resolver,
        tagstore=ts,
        dest_q=q,
        track_suffix="",
        do_retry=False,
    )
    new_tags = [
        get_tag(model_track="asr", text="new tag 1", source_media="source1"),
        get_tag(model_track="asr", text="new tag 2", source_media="source1"),
    ]
    second_session.upload_tags(tags=new_tags, tagged_sources=["source1"])

    # source1's old tags are replaced; source2 was not reprocessed so it is untouched
    assert {t.text for t in ts.find_tags(q=q, sources=["source1"])} == {"new tag 1", "new tag 2"}
    assert {t.text for t in ts.find_tags(q=q, sources=["source2"])} == {"old tag 3"}


def test_tags_survive_when_progress_lags_across_ticks(upload_session, get_tag):
    """A source's tags on one track can be posted a tick before its progress arrives
    (the tagger reads tags and progress independently). The later tick, once the
    source shows up in progress, must not delete the tags posted earlier."""
    ts = upload_session.tagstore
    q = upload_session.dest_q

    # tick 1: speech_to_text tags for source1 arrive, but no progress yet
    upload_session.upload_tags(
        tags=[get_tag(model_track="", text="word", source_media="source1")],
        tagged_sources=[],
    )
    # tick 2: auto_captions tags for source1 arrive together with its progress
    upload_session.upload_tags(
        tags=[get_tag(model_track="auto_captions", text="sentence", source_media="source1")],
        tagged_sources=["source1"],
    )

    assert {t.text for t in ts.find_tags(q=q, track="speech_to_text")} == {"word"}
    assert {t.text for t in ts.find_tags(q=q, track="auto_captions")} == {"sentence"}


def test_two_tracks_two_sources_with_batched_progress(upload_session, get_tag):
    """Mirror the asr container: word tags (default track) for two sources land before
    progress, then sentence tags (auto_captions) land with batched progress for both."""
    ts = upload_session.tagstore
    q = upload_session.dest_q

    upload_session.upload_tags(
        tags=[
            get_tag(model_track="", text="word1", source_media="source1"),
            get_tag(model_track="", text="word2", source_media="source2"),
        ],
        tagged_sources=[],
    )
    upload_session.upload_tags(
        tags=[
            get_tag(model_track="auto_captions", text="sentence1", source_media="source1"),
            get_tag(model_track="auto_captions", text="sentence2", source_media="source2"),
        ],
        tagged_sources=["source1", "source2"],
    )

    assert {t.text for t in ts.find_tags(q=q, track="speech_to_text")} == {"word1", "word2"}
    assert {t.text for t in ts.find_tags(q=q, track="auto_captions")} == {"sentence1", "sentence2"}


def test_incremental_tags_for_same_pair_are_not_wiped(upload_session, get_tag):
    """Tags for the same (track, source) streaming across ticks should append, not
    delete what was already posted."""
    ts = upload_session.tagstore
    q = upload_session.dest_q

    upload_session.upload_tags(
        tags=[get_tag(model_track="", text="first", source_media="source1")],
        tagged_sources=["source1"],
    )
    upload_session.upload_tags(
        tags=[get_tag(model_track="", text="second", source_media="source1")],
        tagged_sources=["source1"],
    )

    assert {t.text for t in ts.find_tags(q=q, track="speech_to_text")} == {"first", "second"}


def test_delete_by_source_is_scoped_to_model(upload_session, track_resolver, get_tag):
    """Two sessions for different models tagging the same source must not clobber
    each other when deleting pre-existing tags before posting. Deletion is scoped
    to the model, so a different model's tags for the same source are untouched."""
    ts = upload_session.tagstore
    q = upload_session.dest_q

    # session A (model "asr") tags source1 on track_a
    upload_session.upload_tags(
        tags=[get_tag(model_track="track_a", text="A tag", source_media="source1")],
        tagged_sources=["source1"],
    )

    # a session for a DIFFERENT model tags the SAME source
    session_b = UploadSession(
        feature="other_model",
        track_resolver=track_resolver,
        tagstore=ts,
        dest_q=q,
        track_suffix="",
        do_retry=False,
    )
    session_b.upload_tags(
        tags=[get_tag(model_track="track_b", text="B tag", source_media="source1")],
        tagged_sources=["source1"],
    )

    # session B's pre-post delete (scoped to model "other_model") must not have
    # wiped the "asr" model's track_a tag for source1
    assert {t.text for t in ts.find_tags(q=q, track="track_a")} == {"A tag"}
    assert {t.text for t in ts.find_tags(q=q, track="track_b")} == {"B tag"}


def test_reupload_replaces_tags_no_model_track(upload_session, track_resolver, get_tag):
    """Tags without a model_track resolve to the feature's configured track, and a
    re-run should still replace prior tags for the reprocessed source."""
    ts = upload_session.tagstore
    q = upload_session.dest_q

    upload_session.upload_tags(
        tags=[get_tag(model_track="", text="old tag", source_media="source1")],
        tagged_sources=["source1"],
    )
    # asr's first configured track is speech_to_text
    assert {t.text for t in ts.find_tags(q=q, track="speech_to_text")} == {"old tag"}

    second_session = UploadSession(
        feature="asr",
        track_resolver=track_resolver,
        tagstore=ts,
        dest_q=q,
        track_suffix="",
        do_retry=False,
    )
    second_session.upload_tags(
        tags=[get_tag(model_track="", text="new tag", source_media="source1")],
        tagged_sources=["source1"],
    )
    assert {t.text for t in ts.find_tags(q=q, track="speech_to_text")} == {"new tag"}


def test_processed_sources_without_tags_clears_configured_tracks(upload_session, track_resolver, get_tag):
    """A source can be processed without producing any tags; its pre-existing tags on
    the feature's configured tracks should still be cleared."""
    ts = upload_session.tagstore
    q = upload_session.dest_q

    # a prior run left tags for source1 on the configured speech_to_text track
    upload_session.upload_tags(
        tags=[get_tag(model_track="", text="stale tag", source_media="source1")],
        tagged_sources=["source1"],
    )
    assert {t.text for t in ts.find_tags(q=q, track="speech_to_text")} == {"stale tag"}

    # a fresh run processes source1 but the container produces no tags for it
    second_session = UploadSession(
        feature="asr",
        track_resolver=track_resolver,
        tagstore=ts,
        dest_q=q,
        track_suffix="",
        do_retry=False,
    )
    second_session.upload_tags(tags=[], tagged_sources=["source1"])

    # the stale tags are cleared even though no new tags were produced
    assert ts.find_tags(q=q, track="speech_to_text") == []


def test_configured_track_deletion_respects_suffix(track_resolver, mock_q, filesystem_tagstore, get_tag):
    """The track suffix must be applied when clearing configured tracks."""
    ts = filesystem_tagstore
    q = mock_q

    session = UploadSession(
        feature="asr",
        track_resolver=track_resolver,
        tagstore=ts,
        dest_q=q,
        track_suffix="v2",
        do_retry=False,
    )
    session.upload_tags(
        tags=[get_tag(model_track="", text="stale tag", source_media="source1"),
              get_tag(model_track="another_track", text="stale tag 2", source_media="source2")],
        tagged_sources=["source1"],
    )
    assert {t.text for t in ts.find_tags(q=q, track="speech_to_text_v2")} == {"stale tag"}
    assert {t.text for t in ts.find_tags(q=q, track="another_track_v2")} == {"stale tag 2"}

    # a new session over the same suffixed track processes source1 with no tags
    session2 = UploadSession(
        feature="asr",
        track_resolver=track_resolver,
        tagstore=ts,
        dest_q=q,
        track_suffix="v2",
        do_retry=False,
    )
    session2.upload_tags(tags=[], tagged_sources=["source1"])

    assert ts.find_tags(q=q, track="speech_to_text_v2") == []
    # we gave another source to this one
    assert len(ts.find_tags(q=q, track="another_track_v2")) == 1

    # check that we can upload a new tag and it will clear the old tag, even without passing in tagged_sources
    session2.upload_tags(tags=[get_tag(model_track="another_track", text="new tag 2", source_media="source2")], tagged_sources=[])
    assert {t.text for t in ts.find_tags(q=q, track="another_track_v2")} == {"new tag 2"}

    # check that even after marking as tagged_source we still get back the previous tag (i.e the delete by batch happens during the first time the source is seen)
    session2.upload_tags(tags=[], tagged_sources=["source2"])
    assert {t.text for t in ts.find_tags(q=q, track="another_track_v2")} == {"new tag 2"}
    assert ts.find_tags(q=q, track="speech_to_text_v2") == []

def test_retry_on_upload_failure(upload_session, get_tag):
    upload_session.retry = True
    tags = [
        get_tag(model_track="asr", text="hello world", source_media="/path/to/source1.mp4"),    
    ]

    tagged_sources = ["source1"]

    original_fn = upload_session.tagstore.upload_tags

    upload_session.tagstore.upload_tags = Mock(side_effect=Exception("upload failed"))

    upload_session.upload_tags(tags=tags, tagged_sources=tagged_sources)
    
    uploaded_sources = upload_session.get_uploaded_sources()
    assert not uploaded_sources

    # Restore the original function
    upload_session.tagstore.upload_tags = original_fn

    # Retry should allow subsequent uploads to succeed
    upload_session.upload_tags(tags=tags, tagged_sources=tagged_sources)
    uploaded_sources = upload_session.get_uploaded_sources()
    assert set(uploaded_sources) == {"source1"}