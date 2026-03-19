import math
from unittest.mock import MagicMock

from src.fetch.model import VideoMetadata
from src.fetch.impl.processors import SkipWorker
from src.fetch.model import TimeRangeScope

def test_skip_worker(temp_dir: str) -> None:
    """Test SkipWorker functionality"""

    part_duration = 10
    num_parts = 12
    start_time = 35
    chunk_size = 10

    fake_q = MagicMock(qid="test_qhit", token="test_token")

    worker = SkipWorker(
        q=fake_q,
        scope=TimeRangeScope(
            start_time=start_time,
            end_time=None,
            chunk_size=chunk_size,
            stream="video"
        ),
        meta=VideoMetadata(
            part_duration=part_duration,
            parts=[f"part_{i}" for i in range(num_parts)],
            fps=30,
            codec_type="video"
        ),
        ignore_sources=["0000020000_0000030000", "0000040000_0000050000"],
        output_dir=temp_dir
    )

    result = worker.download()

    assert len(result.sources) == math.ceil(part_duration * num_parts / chunk_size) - 4 # 2 from ignored list, 3 from before start_time, minus intersection
    assert len(result.failed) == 0
    source_names = [source.name for source in result.sources]
    assert "0000000000_0000010000" not in source_names
    assert "0000010000_0000020000" not in source_names
    assert "0000020000_0000030000" not in source_names
    assert "0000040000_0000050000" not in source_names
    assert "0000030000_0000040000" in source_names
    assert "0000050000_0000060000" in source_names