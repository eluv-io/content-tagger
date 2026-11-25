from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.fetch.impl.live import LiveWorker
from src.fetch.model import LiveScope, MediaMetadata
from src.fetch.rate_limit import FetchRateLimiter

def fake_segment(object_id, dest_path, segment_idx, segment_length, stream):
    open(dest_path, "wb").close()
    return SimpleNamespace(
        seg_num=segment_idx,
        seg_offset_millis=segment_idx * segment_length * 1000,
        actual_duration=segment_length,
        seg_time_epoch_millis=0,
    )

@pytest.fixture(autouse=True)
def no_center_segment(monkeypatch):
    monkeypatch.setattr("src.fetch.impl.live.center_segment", lambda path: None)

def make_worker(temp_dir: str, ignore_sources: list[str] = [], segment_length: int = 4, max_duration: int | None = 20, start_at_live_edge: bool = False, fps: float = 30, keyint: int | None = 60) -> LiveWorker:
    qapi = MagicMock()
    qapi.id.return_value = "iq__fake"
    qapi.live_media_segment.side_effect = fake_segment
    return LiveWorker(
        qapi=qapi,
        scope=LiveScope(stream="video", segment_length=segment_length, max_duration=max_duration, start_at_live_edge=start_at_live_edge),
        rate_limiter=FetchRateLimiter(max_concurrent=1),
        meta=MediaMetadata(sources=[], fps=fps),
        keyint=keyint,
        ignore_sources=ignore_sources,
        output_dir=temp_dir,
    )

def test_live_worker_incremental_segments(temp_dir: str):
    worker = make_worker(temp_dir)

    sources = []
    result = None
    for _ in range(10):
        result = worker.download()
        assert len(result.sources) <= 1
        sources.extend(result.sources)
        if result.done:
            break
    assert result
    assert result.done
    assert [s.name for s in sources] == [f"video:segment_4_{i}" for i in range(5)]
    assert all(s.offset < 20 * 1000 for s in sources)

def test_live_worker_respects_ignore_sources(temp_dir: str):
    worker = make_worker(temp_dir, ignore_sources=["video:segment_4_0", "video:segment_4_1"])

    assert [s.name for s in worker.download().sources] == ["video:segment_4_2"]
    assert [s.name for s in worker.download().sources] == ["video:segment_4_3"]

def test_live_worker_all_ignored(temp_dir: str):
    worker = make_worker(temp_dir, ignore_sources=[f"video:segment_4_{i}" for i in range(5)])

    result = worker.download()
    assert len(result.sources) == 0
    assert result.done is True

def test_live_worker_metadata(temp_dir: str):
    worker = make_worker(temp_dir)

    assert len(worker.metadata().sources) == 0
    worker.download()
    assert len(worker.metadata().sources) == 1
    worker.download()
    assert len(worker.metadata().sources) == 2

def test_live_worker_exit(temp_dir: str):
    worker = make_worker(temp_dir)
    worker.exit = MagicMock(is_set=lambda: True)

    result = worker.download()
    assert result.sources == []
    assert result.done is True

def make_edge_worker(temp_dir: str, max_duration: int | None = None, frames_written: int = 9015, **kwargs) -> LiveWorker:
    worker = make_worker(temp_dir, start_at_live_edge=True, max_duration=max_duration, **kwargs)
    worker.qapi.live_status.return_value = {"custom": {"status": {"video": {"first_key_frame_pts": 0, "frames_written": {"total_frames_written": frames_written}}}}}
    return worker

def test_live_worker_starts_at_live_edge(temp_dir: str):
    worker = make_edge_worker(temp_dir)

    # 9015 frames / 30fps = 300.5s -> segment 75
    assert [s.name for s in worker.download().sources] == ["video:segment_4_75"]
    assert [s.name for s in worker.download().sources] == ["video:segment_4_76"]

def test_live_edge_uses_keyint_aligned_segments(temp_dir: str):
    # 59.94fps with keyint 120: fabric serves 4s requests as 240 frames (4.004s)
    # ~16.8h in, seconds-based math (60642.7s // 4 = 15160) would land ~15 segments past the edge
    worker = make_edge_worker(temp_dir, fps=60000 / 1001, keyint=120, frames_written=3634860)
    assert worker._live_edge_idx() == 3634860 // 240

def test_live_edge_rounds_up_to_keyint(temp_dir: str):
    # 3s at 59.94fps = 179.82 frames -> rounded up to 240 (2 keyframe intervals)
    worker = make_edge_worker(temp_dir, segment_length=3, fps=60000 / 1001, keyint=120, frames_written=24000)
    assert worker._live_edge_idx() == 100

def test_live_edge_without_keyint_backs_off(temp_dir: str):
    # 4s at 30fps = 120 frames -> 9015 // 120 = 75, minus 2 for safety
    worker = make_edge_worker(temp_dir, keyint=None)
    assert worker._live_edge_idx() == 73
