

import os
import tempfile
import subprocess

def center_segment(segment_path: str) -> None:
    """Set pks_pts times to start at 0 for live segments."""
    dir_ = os.path.dirname(segment_path)

    with tempfile.NamedTemporaryFile(dir=dir_, delete=False, suffix=".mp4") as tmp:
        tmp_path = tmp.name

    subprocess.run(
        [
            "ffmpeg",
            "-i", segment_path,
            "-y",
            "-map", "0",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            tmp_path
        ],
        check=True,
        capture_output=True
    )

    os.replace(tmp_path, segment_path)