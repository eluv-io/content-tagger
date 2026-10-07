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

def make_worker(temp_dir: str, ignore_sources: list[str] = [], segment_length: int = 4, max_duration: int = 20) -> LiveWorker:
    qapi = MagicMock()
    qapi.id.return_value = "iq__fake"
    qapi.live_media_segment.side_effect = fake_segment
    return LiveWorker(
        qapi=qapi,
        scope=LiveScope(stream="video", segment_length=segment_length, max_duration=max_duration),
        rate_limiter=FetchRateLimiter(max_concurrent=1),
        meta=MediaMetadata(sources=[], fps=30),
        ignore_sources=ignore_sources,
        output_dir=temp_dir,
    )

def test_live_worker_incremental_segments(temp_dir: str):
    worker = make_worker(temp_dir)

    sources = []
    for _ in range(10):
        result = worker.download()
        assert len(result.sources) <= 1
        sources.extend(result.sources)
        if result.done:
            break

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

def test_live_worker_fps_from_first_segment(temp_dir: str, monkeypatch):
    monkeypatch.setattr("src.fetch.impl.live.get_fps", lambda path: 25.0)
    worker = make_worker(temp_dir)
    worker.meta.fps = None

    assert worker.metadata().fps is None
    worker.download()
    assert worker.metadata().fps == 25.0
