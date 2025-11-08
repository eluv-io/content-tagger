import pytest
import os

from src.fetch.factory import FetchFactory
from src.fetch.model import AssetScope, DownloadRequest, FetcherConfig, VideoScope, LiveScope
from src.tags.tagstore.filesystem_tagstore import Tag
from src.common.content import Content
from src.common.errors import MissingResourceError

VOD_QHIT = "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm"
LEGACY_VOD_QHIT = "iq__cebzuQ8BqsWZyoUdnTXCe23fUgz"
ASSETS_QHIT = "iq__cebzuQ8BqsWZyoUdnTXCe23fUgz"

@pytest.fixture
def fetcher(fetcher_config: FetcherConfig, tag_store) -> FetchFactory:
    """Create a FetchFactory instance for testing"""
    return FetchFactory(config=fetcher_config, ts=tag_store)

@pytest.fixture
def fetcher_config(temp_dir: str) -> FetcherConfig:
    """Create a FetcherConfig for testing"""
    parts_path = os.path.join(temp_dir, "parts")
    os.makedirs(parts_path, exist_ok=True)
    
    return FetcherConfig(
        author="tagger",
        max_downloads=2
    )

@pytest.fixture
def legacy_vod_content(qfactory, tag_store) -> Content:
    auth_token = os.getenv("TEST_AUTH")
    
    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")
    
    try:
        q = qfactory.create_content(qhit=LEGACY_VOD_QHIT, auth=auth_token)
    except Exception as e:
        pytest.skip(f"Failed to create legacy VOD content: {e}")

    batch_ids = tag_store.find_batches(q=q)
    for batch_id in batch_ids:
        tag_store.delete_batch(batch_id, q=q)

    return q


@pytest.fixture
def vod_content(qfactory, tag_store) -> Content:
    auth_token = os.getenv("TEST_AUTH")
    
    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")
    
    try:
        q = qfactory.create_content(qhit=VOD_QHIT, auth=auth_token)
    except Exception as e:
        pytest.skip(f"Failed to create modern VOD content: {e}")

    batch_ids = tag_store.find_batches(q=q)
    for batch_id in batch_ids:
        tag_store.delete_batch(batch_id, q=q)

    return q

@pytest.fixture
def assets_content(qfactory, tag_store) -> Content:
    auth_token = os.getenv("TEST_AUTH")

    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")

    try:
        q = qfactory.create_content(qhit=ASSETS_QHIT, auth=auth_token)
    except Exception as e:
        pytest.skip(f"Failed to create assets content: {e}")

    batch_ids = tag_store.find_batches(q=q)
    for batch_id in batch_ids:
        tag_store.delete_batch(batch_id, q=q)

    return q


@pytest.mark.parametrize("content_fixture", ["vod_content", "legacy_vod_content"])
def test_download_with_replace_true(
    fetcher: FetchFactory, 
    request,
    content_fixture: str,
    media_dir: str
):
    """Test downloading with replace=True using different content fixtures"""
    vod_content = request.getfixturevalue(content_fixture)

    print(vod_content.qhit)

    """Test downloading with replace=True"""
    req = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=60
        ),
        output_dir=media_dir,
        preserve_track="track"
    )
    
    # Download once
    worker = fetcher.get_session(vod_content, req)
    result1 = worker.download()
    assert len(result1.sources) == 2
    assert len(result1.failed) == 0

    tagstore = fetcher.ts

    job = tagstore.create_batch(
        qhit=vod_content.qid,
        stream="video",
        author="tagger",
        track="track",
        q=vod_content
    )

    first_source = result1.sources[0].name

    tag = Tag(
        start_time=0,
        end_time=1,
        text="hello",
        frame_tags={},
        additional_info={},
        source=first_source,
        batch_id=job.id
    )

    tagstore.upload_tags([tag], job.id, q=vod_content)

    req2 = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=60
        ),
        output_dir=media_dir,
        preserve_track="track"
    )
    
    worker2 = fetcher.get_session(vod_content, req2)
    result2 = worker2.download()
    assert len(result2.sources) == 1
    assert len(result2.failed) == 0

    req3 = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=60
        ),
        output_dir=media_dir,
        preserve_track="track2"
    )

    # shouldn't replace track2
    worker3 = fetcher.get_session(vod_content, req3)
    result3 = worker3.download()
    assert len(result3.sources) == 2
    assert len(result3.failed) == 0

    # don't set preserve_track
    req4 = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=60
        ),
        output_dir=media_dir,
        preserve_track=""
    )

    worker4 = fetcher.get_session(vod_content, req4)
    result4 = worker4.download()
    assert len(result4.sources) == 2
    assert len(result4.failed) == 0

    # audio instead
    req5 = DownloadRequest(
        scope=VideoScope(
            stream="audio",
            start_time=0,
            end_time=60
        ),
        output_dir=os.path.join(media_dir, "audio"),
        preserve_track="track"
    )

    try:
        worker5 = fetcher.get_session(vod_content, req5)
        result5 = worker5.download()
        assert len(result5.sources) == 2
        assert len(result5.failed) == 0
    except MissingResourceError:
        pass

def test_fetch_assets_with_preserve_track(
    fetcher: FetchFactory, 
    assets_content: Content,
    media_dir: str
):
    req1 = DownloadRequest(
        scope=AssetScope(
            assets=None
        ),
        output_dir=media_dir,
        preserve_track=""
    )

    worker1 = fetcher.get_session(assets_content, req1)
    result1 = worker1.download()
    assert len(result1.sources) > 0, "Should have downloaded some assets"
    assert len(result1.failed) == 0, "Should have no failed downloads initially"
    
    all_asset_names = [source.name for source in result1.sources]
    selected_assets = all_asset_names[:3]
    
    req2 = DownloadRequest(
        scope=AssetScope(
            assets=selected_assets
        ),
        output_dir=media_dir,
        preserve_track=""
    )

    worker2 = fetcher.get_session(assets_content, req2)
    result2 = worker2.download()
    assert len(result2.sources) == len(selected_assets), "Should return all requested assets"
    assert len(result2.failed) == 0, "Should have no failed downloads"
    
    # Verify the returned assets match what we requested
    returned_asset_names = [source.name for source in result2.sources]
    assert set(returned_asset_names) == set(selected_assets), "Returned assets should match requested assets"
    
    # Third test: Add tags for some assets and test preserve_track functionality
    tagstore = fetcher.ts
    
    job = tagstore.create_batch(
        qhit=assets_content.qhit,
        stream="assets",
        author=fetcher.config.author,
        track="asset_track",
        q=assets_content
    )
    batch_id = job.id
    
    # Tag the first two assets
    assets_to_tag = selected_assets[:2] if len(selected_assets) >= 2 else selected_assets
    tags = []
    for asset_name in assets_to_tag:
        tag = Tag(
            start_time=0,
            end_time=1,
            text="test asset tag",
            frame_tags={},
            additional_info={},
            source=asset_name,
            batch_id=batch_id
        )
        tags.append(tag)
    
    tagstore.upload_tags(tags, batch_id, q=assets_content)
    
    req3 = DownloadRequest(
        scope=AssetScope(
            assets=selected_assets
        ),
        output_dir=media_dir,
        preserve_track="asset_track"
    )

    worker3 = fetcher.get_session(assets_content, req3)
    result3 = worker3.download()
    returned_asset_names_after_tagging = [source.name for source in result3.sources]
    
    for tagged_asset in assets_to_tag:
        assert tagged_asset not in returned_asset_names_after_tagging, f"Tagged asset {tagged_asset} should be excluded when preserve_track is set"
    
    # Should still return untagged assets
    untagged_assets = [asset for asset in selected_assets if asset not in assets_to_tag]
    for untagged_asset in untagged_assets:
        assert untagged_asset in returned_asset_names_after_tagging, f"Untagged asset {untagged_asset} should still be returned"
    
    # Fifth test: fetch assets with different preserve_track - should return all assets
    req4 = DownloadRequest(
        scope=AssetScope(
            assets=selected_assets
        ),
        output_dir=media_dir,
        preserve_track="different_track"
    )

    worker4 = fetcher.get_session(assets_content, req4)
    result4 = worker4.download()
    returned_asset_names_different_track = [source.name for source in result4.sources]
    assert set(returned_asset_names_different_track) == set(selected_assets), "Should return all assets when preserve_track doesn't match"
    
    # Upload tags to selected tags with author="user" and track="another track", check that downloading again
    # returns all the selected assets (tagger author is special)

    new_job = tagstore.create_batch(
        qhit=assets_content.qhit,
        stream="assets",
        author="user",
        track="another_track",
        q=assets_content
    )

    newtags = []
    for asset in selected_assets:
        tag = Tag(
            start_time=0,
            end_time=1,
            text="asset_track",
            frame_tags={},
            additional_info={},
            source=asset,
            batch_id=new_job.id
        )
        newtags.append(tag)
    tagstore.upload_tags(newtags, new_job.id, q=assets_content)

    # Verify that the tags were uploaded correctly
    uploaded_tags = tagstore.find_tags(batch_id=new_job.id, q=assets_content)
    assert len(uploaded_tags) == len(newtags), "Not all tags were uploaded"
    for tag in newtags:
        assert tag in uploaded_tags, f"Tag {tag} was not found in uploaded tags"

    req5 = DownloadRequest(
        scope=AssetScope(
            assets=selected_assets
        ),
        output_dir=media_dir,
        preserve_track="asset_track"
    )

    worker5 = fetcher.get_session(assets_content, req5)
    result5 = worker5.download()
    returned_asset_names_after_tagging = [source.name for source in result5.sources]

    assert len(returned_asset_names_after_tagging) >= 1

    for assetname in returned_asset_names_after_tagging:
        assert assetname in selected_assets, f"Asset {assetname} should be in the selected assets after tagging"
        # tagged before
        assert assetname not in assets_to_tag
        assert assetname in untagged_assets

def test_live_worker_incremental_segments(
    fetcher: FetchFactory,
    live_q: Content,
    media_dir: str
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
        output_dir=media_dir,
        preserve_track=""
    )
    
    worker = fetcher.get_session(live_q, req)
    
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
            assert len(name_parts) == 3, f"Expected name format 'segment_<size>_<idx>', got {source.name}"
            assert name_parts[0] == "segment", f"Expected name to start with 'segment', got {source.name}"
            
            chunk_size_from_name = int(name_parts[1])
            seg_idx = int(name_parts[2])

            assert seg_idx == last_idx +1
            last_idx = seg_idx
            
            assert chunk_size_from_name == chunk_size, f"Expected chunk size {chunk_size}, got {chunk_size_from_name}"
            
            # Check offset is reasonable
            assert source.offset >= 0, f"Offset should be non-negative, got {source.offset}"

            assert source.wall_clock is not None
            assert source.wall_clock >= last_wall_clock
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