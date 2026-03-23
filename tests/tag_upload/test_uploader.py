

from dataclasses import asdict, dataclass
from unittest.mock import Mock

import pytest

from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs, TagContentStatusReport
from src.tagging.uploading.uploader import UploadSession
from src.tags.track_resolver import TrackArgs, TrackResolver, TrackResolverConfig


def test_upload_tags(upload_session, get_tag):
    tags = [
        get_tag(model_track="asr", text="hello world"),
        get_tag(model_track="caption", text="test tag", start_time=100, end_time=200)
    ]

    upload_session.upload_tags(tags=tags, tagged_sources=[t.source_media for t in tags])

    ts = upload_session.tagstore
    track = ts.get_track(name="speech_to_text", q=upload_session.dest_q)

    # check that tracks exist
    assert track is not None
    assert track.name == "speech_to_text"
    assert track.label == "Speech to Text"

    track = ts.get_track(name="object_detection", q=upload_session.dest_q)
    assert track is not None
    assert track.name == "object_detection"
    assert track.label == "Object Detection"

    ts_tags = ts.find_tags(q=upload_session.dest_q)
    assert len(ts_tags) == 2

    speech_tag = ts.find_tags(q=upload_session.dest_q, track="speech_to_text")[0]
    assert speech_tag.text == "hello world"

    caption_tag = ts.find_tags(q=upload_session.dest_q, track="object_detection")[0]
    assert caption_tag.text == "test tag"
    assert caption_tag.start_time == 100
    assert caption_tag.end_time == 200

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
    batch = upload_session._get_batch(model_track="asr")

    assert batch is not None

    db_batch = ts.get_batch(batch_id=batch, q=upload_session.dest_q)
    assert db_batch is not None
    assert db_batch.additional_info == {"tagger": asdict(report)}

def test_get_uploaded_sources(upload_session, get_tag):
    tags = [
        get_tag(model_track="asr", text="hello world", source_media="/path/to/source1.mp4"),
        get_tag(model_track="caption", text="test tag", start_time=100, end_time=200, source_media="/path/to/source2.mp4")
    ]

    tagged_sources = ["source1", "source2"]

    upload_session.upload_tags(tags=tags, tagged_sources=tagged_sources)

    uploaded_sources = upload_session.get_uploaded_sources()

    assert set(uploaded_sources) == {"source1", "source2"}