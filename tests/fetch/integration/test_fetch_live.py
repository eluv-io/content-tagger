    

from src.common.content import Content
from src.fetch.factory import FetchFactory
from src.fetch.impl.live import LiveWorker
from src.fetch.model import DownloadRequest, LiveScope


def test_live_worker_incremental_segments(
    fetcher: FetchFactory,
    q_live: Content,
    temp_dir: str
):
    """Test LiveWorker returns incremental segments and respects max_duration"""
    
    max_duration = 20  # 20 seconds
    chunk_size = 4  # 4 second chunks
    
    req = DownloadRequest(
        scope=LiveScope(
            stream="video",
            chunk_size=chunk_size,
            max_duration=max_duration
        ),
        output_dir=temp_dir,
        ignore_sources=[],
    )
    
    worker = fetcher.get_session(q_live, req)
    
    # Collect all segments
    all_sources = []
    segment_indices = []
    done = False
    call_count = 0
    max_calls = 10  # Safety limit

    last_idx = -1
    last_wall_clock = 0
    
    while not done and call_count < max_calls:
        result = worker.download()
        call_count += 1
        
        # Should return exactly one source per call (or empty when waiting)
        assert len(result.sources) <= 1, f"Expected at most 1 source, got {len(result.sources)}"
        
        if len(result.sources) == 1:
            source = result.sources[0]
            all_sources.append(source)
            
            # Extract segment index from name like "segment_4_0"
            name_parts = source.name.split('_')
            assert len(name_parts) == 3
            
            chunk_size_from_name = int(name_parts[1])
            seg_idx = int(name_parts[2])

            assert seg_idx == last_idx +1
            last_idx = seg_idx
            
            assert chunk_size_from_name == chunk_size, f"Expected chunk size {chunk_size}, got {chunk_size_from_name}"
            
            # Check offset is reasonable
            assert source.offset >= 0, f"Offset should be non-negative, got {source.offset}"

            assert source.wall_clock is not None
            assert source.wall_clock > last_wall_clock
            last_wall_clock = source.wall_clock
            
            print(f"Call {call_count}: Got segment {seg_idx} at offset {source.offset}s")
        
        done = result.done
        
        if done:
            # When done, check that we stopped because of max_duration
            if len(all_sources) > 0:
                last_source = all_sources[-1]
                assert last_source.offset <= max_duration * 1000, \
                    f"Last segment offset {last_source.offset} should be less than max_duration {max_duration}"
            break
    
    # Verify we got some segments
    assert len(all_sources) > 0, "Should have downloaded at least one segment"
    
    # Verify indices are incrementing
    assert segment_indices == sorted(segment_indices), \
        f"Segment indices should be in ascending order, got {segment_indices}"
    
    # Verify indices are consecutive
    for i in range(len(segment_indices) - 1):
        assert segment_indices[i+1] == segment_indices[i] + 1, \
            f"Segment indices should be consecutive, got gap between {segment_indices[i]} and {segment_indices[i+1]}"
    
    # Verify we stopped before max_duration
    assert done, "Worker should have set done=True when reaching max_duration"
    
    # Calculate expected number of segments
    # Should be roughly max_duration / chunk_size
    expected_segments = max_duration / chunk_size
    assert len(all_sources) >= expected_segments * 0.8, \
        f"Expected at least {expected_segments * 0.8:.0f} segments, got {len(all_sources)}"
    assert len(all_sources) <= expected_segments * 1.2, \
        f"Expected at most {expected_segments * 1.2:.0f} segments, got {len(all_sources)}"
    
    print(f"LiveWorker test completed: {len(all_sources)} segments downloaded in {call_count} calls")

def test_live_worker_respects_ignore_sources(
    fetcher: FetchFactory,
    q_live: Content,
    temp_dir: str
):
    """LiveWorker should skip source names provided in ignore_sources."""
    chunk_size = 4
    ignored = [
        f"video:segment_{chunk_size}_0",
        f"video:segment_{chunk_size}_1",
    ]

    req = DownloadRequest(
        scope=LiveScope(stream="video", chunk_size=chunk_size, max_duration=20),
        output_dir=temp_dir,
        ignore_sources=ignored,
    )

    worker = fetcher.get_session(q_live, req)
    assert isinstance(worker, LiveWorker)
    worker.ignore_sources = set(ignored)

    result = worker.download()
    assert len(result.sources) == 1

    first_source = result.sources[0]
    assert first_source.name == f"video:segment_{chunk_size}_2"

    result = worker.download()
    assert len(result.sources) == 1
    assert result.sources[0].name == f"video:segment_{chunk_size}_3"

    result = worker.download()
    assert len(result.sources) == 0
    assert result.done is True

def test_live_worker_all_ignored(
    fetcher: FetchFactory,
    q_live: Content,
    temp_dir: str
):
    """LiveWorker should skip source names provided in ignore_sources."""
    chunk_size = 4
    ignored = [
        f"video:segment_{chunk_size}_0",
        f"video:segment_{chunk_size}_1",
        f"video:segment_{chunk_size}_2",
        f"video:segment_{chunk_size}_3",
    ]

    req = DownloadRequest(
        scope=LiveScope(stream="video", chunk_size=chunk_size, max_duration=20),
        output_dir=temp_dir,
        ignore_sources=ignored,
    )

    worker = fetcher.get_session(q_live, req)
    assert isinstance(worker, LiveWorker)
    worker.ignore_sources = set(ignored)

    result = worker.download()
    assert len(result.sources) == 0
    assert result.done is True

def test_live_worker_metadata(
    fetcher: FetchFactory,
    q_live: Content,
    temp_dir: str
):
    req = DownloadRequest(
        scope=LiveScope(stream="video", chunk_size=4, max_duration=20),
        output_dir=temp_dir,
        ignore_sources=[],
    )

    worker = fetcher.get_session(q_live, req)
    assert len(worker.metadata().sources) == 0
    worker.download()
    assert len(worker.metadata().sources) == 1
    worker.download()
    assert len(worker.metadata().sources) == 2