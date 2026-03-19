
import pytest

from src.tag_containers.model import ModelTag


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