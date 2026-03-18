import pytest

from src.tag_containers.model import ModelTag
from src.fetch.model import Source
from src.tagging.uploading.align import align_tags

def test_align_tags():
    tags = [
        ModelTag(
            start_time=1000,
            end_time=2000,
            text="tag1",
            source_media="/path/to/source1.mp4",
            frame_info={"frame_idx": 30},
            model_track="track1",
        ),
        ModelTag(
            start_time=2000,
            end_time=3000,
            text="tag1",
            source_media="/path/to/source2.mp4",
            frame_info={"frame_idx": 60},
            model_track="track1",
        )
    ]

    sources = [
        Source(
            filepath="/path/to/source1.mp4",
            offset=5000,
            wall_clock=100000,
            name="source1",
        ),
        Source(
            filepath="/path/to/source2.mp4",
            offset=10000,
            wall_clock=105000,
            name="source2",
        )
    ]

    aligned = align_tags(tags, sources, fps=30)
    assert len(aligned) == 2

    # check start/end time adjusted by offset
    assert aligned[0].start_time == 6000
    assert aligned[0].end_time == 7000
    # check source renamed
    assert aligned[0].source_media == "source1"
    assert aligned[0].frame_info
    assert aligned[0].frame_info["frame_idx"] == 180
    assert aligned[0].additional_info
    assert aligned[0].additional_info["timestamp_ms"] == 101000

    assert aligned[1].start_time == 12000
    assert aligned[1].end_time == 13000
    assert aligned[1].source_media == "source2"
    assert aligned[1].frame_info
    assert aligned[1].frame_info["frame_idx"] == 360
    assert aligned[1].additional_info
    assert aligned[1].additional_info["timestamp_ms"] == 107000