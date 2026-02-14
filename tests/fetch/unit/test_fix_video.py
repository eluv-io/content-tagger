import os
import subprocess
import tempfile
import shutil

import pytest

from src.fetch.video_process import center_segment

@pytest.fixture
def segment() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "test-data",
        "live_segment.mp4"
    )

def test_center_segment(segment: str) -> None:

    def get_pts_info(file_path: str) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", 
             "-show_entries", "frame=pkt_pts_time", "-of", "csv=p=0", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        pts_times = [float(line) for line in result.stdout.strip().split('\n') if line]
        return pts_times[0]

    original_pts = get_pts_info(segment)
    assert original_pts > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_segment_path = os.path.join(tmpdir, "temp_segment.mp4")
        shutil.copy(segment, temp_segment_path)

        center_segment(temp_segment_path)

        new_pts = get_pts_info(temp_segment_path)
        assert new_pts == 0