
import pytest
import os

from src.fetch.factory import *
from src.fetch.model import *
from src.tags.tagstore.filesystem_tagstore import Tag
from src.common.content import Content
from src.common.errors import MissingResourceError


@pytest.mark.parametrize("content_fixture", ["vod_content_with_tags_clean", "legacy_vod_content_with_tags_clean"])
def test_download_with_replace_true(
    fetcher: FetchFactory, 
    request,
    content_fixture: str,
    temp_dir: str
):
    """Test downloading with replace=True using different content fixtures"""
    vod_content: Content = request.getfixturevalue(content_fixture)

    """Test downloading with replace=True"""
    req = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=60
        ),
        output_dir=temp_dir,
        ignore_sources=[],
    )
    
    # Download once
    worker = fetcher.get_session(vod_content, req)
    result1 = worker.download()
    assert len(result1.sources) == 2
    assert len(result1.failed) == 0

    first_source = result1.sources[0].name

    req2 = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=60
        ),
        output_dir=temp_dir,
        ignore_sources=[first_source],
    )
    
    worker2 = fetcher.get_session(vod_content, req2)
    result2 = worker2.download()
    assert len(result2.sources) == 1
    assert len(result2.failed) == 0

    # audio instead
    req5 = DownloadRequest(
        scope=VideoScope(
            stream="audio",
            start_time=0,
            end_time=60
        ),
        output_dir=os.path.join(temp_dir, "audio"),
        ignore_sources=[],
    )

    try:
        worker5 = fetcher.get_session(vod_content, req5)
        result5 = worker5.download()
        assert len(result5.sources) == 2
        assert len(result5.failed) == 0
    except MissingResourceError:
        pass

def test_fetch_assets(
    fetcher: FetchFactory, 
    assets_content_with_tags_clean: Content,
    temp_dir: str
):
    req1 = DownloadRequest(
        scope=AssetScope(
            assets=None
        ),
        output_dir=temp_dir,
        ignore_sources=[]
    )

    worker1 = fetcher.get_session(assets_content_with_tags_clean, req1)
    result1 = worker1.download()
    assert len(result1.sources) > 0, "Should have downloaded some assets"
    assert len(result1.failed) == 0, "Should have no failed downloads initially"

    num_total_assets = len(worker1.metadata().sources)
    
    all_asset_names = [source.name for source in result1.sources]
    selected_assets = all_asset_names[:3]
    
    req2 = DownloadRequest(
        scope=AssetScope(
            assets=selected_assets
        ),
        output_dir=temp_dir,
        ignore_sources=[],
    )

    worker2 = fetcher.get_session(assets_content_with_tags_clean, req2)
    assert len(worker2.metadata().sources) == len(selected_assets)

    result2 = worker2.download()
    assert len(result2.sources) == len(selected_assets), "Should return all requested assets"
    assert len(result2.failed) == 0, "Should have no failed downloads"
    
    # Verify the returned assets match what we requested
    returned_asset_names = [source.name for source in result2.sources]
    assert set(returned_asset_names) == set(selected_assets), "Returned assets should match requested assets"
    
    # Third test: test ignore_sources
    
    req3 = DownloadRequest(
        scope=AssetScope(
            assets=selected_assets
        ),
        output_dir=temp_dir,
        ignore_sources=[selected_assets[0], selected_assets[1]]
    )

    worker3 = fetcher.get_session(assets_content_with_tags_clean, req3)
    result3 = worker3.download()

    assert len(worker3.metadata().sources) == len(selected_assets) - 2
    
    assert len(result3.sources) == 1, "Should return only one asset after ignoring two"

def test_metadata(
    fetcher: FetchFactory, 
    q,
    temp_dir: str
):

    req = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=1e10
        ),
        output_dir=temp_dir,
        ignore_sources=[],
    )
    
    # Download once
    worker = fetcher.get_session(q, req)
    all_parts = worker.metadata().sources
    assert len(all_parts) > 0

    req2 = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=1e9
        ),
        output_dir=temp_dir,
        ignore_sources=[all_parts[0]],
    )
    
    worker2 = fetcher.get_session(q, req2)
    assert len(worker2.metadata().sources) == len(all_parts) - 1

    req3 = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=60
        ),
        output_dir=os.path.join(temp_dir, "video"),
        ignore_sources=[],
    )

    worker3 = fetcher.get_session(q, req3)
    assert len(worker3.metadata().sources) == 2

def test_incremental(fetcher: FetchFactory, q, temp_dir):
    req = DownloadRequest(
        scope=VideoScope(
            stream="video",
            start_time=0,
            end_time=1e10
        ),
        output_dir=temp_dir,
        ignore_sources=[],
    )

    worker = fetcher.get_session(q, req)

    assert isinstance(worker, VodWorker)
    worker.batch_size = 2

    result1 = worker.download()
    assert len(result1.sources) == 2
    assert not result1.done

    result2 = worker.download()
    assert len(result2.sources) == 2
    assert not result2.done

    for part in result1.sources:
        assert part not in result2.sources
