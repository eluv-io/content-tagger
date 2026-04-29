
import pytest

from src.common.content import Content
from src.tag_containers.model import ModelTag
from src.tagging.uploading.uploader import UploadSession
from src.tags.track_resolver import TrackArgs, TrackResolver, TrackResolverConfig


@pytest.fixture
def get_tag():
    """Fixture to make initializeing ModelTag easier"""
    def fn(
        start_time=0,
        end_time=1000,
        text="test tag",
        source_media="/path/to/source.mp4",
        frame_info=None,
        model_track="test_track",
        additional_info=None,
    ):
        return ModelTag(
            start_time=start_time,
            end_time=end_time,
            text=text,
            source_media=source_media,
            frame_info=frame_info or {},
            model_track=model_track,
            additional_info=additional_info or {},
        )
    return fn

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
        do_retry=False
    )