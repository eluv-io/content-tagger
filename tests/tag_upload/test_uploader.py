

from dataclasses import asdict, dataclass
from unittest.mock import Mock

import pytest

from src.common.content import Content
from src.tagging.fabric_tagging.model import TagArgs, TagContentStatusReport
from src.tagging.uploading.uploader import UploadSession
from src.tags.track_resolver import TrackArgs, TrackResolver, TrackResolverConfig


@pytest.fixture
def mock_q():
    return Content(qid="test_qid", token="")

@pytest.fixture
def track_resolver():
    """Create a simple track resolver for testing"""
    return TrackResolver(cfg=TrackResolverConfig(mapping={
        "caption": TrackArgs(name="object_detection", label="Object Detection"),
        "asr": TrackArgs(name="speech_to_text", label="Speech to Text"),
        "pretty": TrackArgs(name="auto_captions", label="Pretty Speech")
    }))

@pytest.fixture
def upload_session(track_resolver, mock_q, filesystem_tagstore):
    """Create an upload session with the mock track resolver and a mock tagstore"""
    return UploadSession(
        feature="asr",
        track_resolver=track_resolver,
        tagstore=filesystem_tagstore,
        dest_q=mock_q,
    )

def test_upload_tags(upload_session, get_tag):
    tags = [
        get_tag(model_track="asr", text="hello world"),
        get_tag(model_track="caption", text="test tag", start_time=100, end_time=200)
    ]

    upload_session.upload_tags(tags=tags, retry=False)

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

    upload_session.upload_tags(tags=tags, retry=False)

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