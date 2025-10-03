import pytest
import os

from src.fetch.fetch_video import Fetcher
from src.fetch.types import AssetScope, DownloadRequest, FetcherConfig, VideoScope
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore, Tag
from src.common.content import Content
from src.common.errors import MissingResourceError

VOD_QHIT = "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm"
LEGACY_VOD_QHIT = "hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU"
ASSETS_QHIT = "hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU"

@pytest.fixture
def fetcher_config(temp_dir: str) -> FetcherConfig:
    """Create a FetcherConfig for testing"""
    parts_path = os.path.join(temp_dir, "parts")
    os.makedirs(parts_path, exist_ok=True)
    
    return FetcherConfig(
        max_downloads=2,
        parts_dir=parts_path,
        author="tagger"
    )

@pytest.fixture
def fetcher(fetcher_config: FetcherConfig, tag_store: FilesystemTagStore) -> Fetcher:
    """Create a Fetcher instance for testing"""
    return Fetcher(config=fetcher_config, ts=tag_store)

@pytest.fixture
def legacy_vod_content(qfactory) -> Content:
    auth_token = os.getenv("TEST_AUTH")
    
    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")
    
    try:
        return qfactory.create_content(qhit=LEGACY_VOD_QHIT, auth=auth_token)
    except Exception as e:
        pytest.skip(f"Failed to create legacy VOD content: {e}")


@pytest.fixture
def vod_content(qfactory) -> Content:
    auth_token = os.getenv("TEST_AUTH")
    
    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")
    
    try:
        return qfactory.create_content(qhit=VOD_QHIT, auth=auth_token)
    except Exception as e:
        pytest.skip(f"Failed to create modern VOD content: {e}")

@pytest.fixture
def assets_content(qfactory) -> Content:
    auth_token = os.getenv("TEST_AUTH")

    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")

    try:
        return qfactory.create_content(qhit=ASSETS_QHIT, auth=auth_token)
    except Exception as e:
        pytest.skip(f"Failed to create assets content: {e}")


@pytest.mark.parametrize("content_fixture", ["vod_content", "legacy_vod_content"])
def test_download_with_replace_true(
    fetcher: Fetcher, 
    request,
    content_fixture: str
):
    """Test downloading with replace=True using different content fixtures"""
    vod_content = request.getfixturevalue(content_fixture)

    print(vod_content.qhit)

    """Test downloading with replace=True"""
    req = DownloadRequest(
        stream_name="video",
        scope=VideoScope(
            start_time=0,
            end_time=60
        ),
        preserve_track="track"
    )
    
    # Download once
    result1 = fetcher.download(vod_content, req)
    assert len(result1.successful_sources) == 2
    assert len(result1.failed) == 0

    tagstore = fetcher.ts

    job = tagstore.start_job(
        qhit=vod_content.qid,
        stream="video",
        author="tagger",
        track="track",
        q=vod_content
    )

    first_source = result1.successful_sources[0].name

    tag = Tag(
        start_time=0,
        end_time=1,
        text="hello",
        additional_info={},
        source=first_source,
        jobid=job.id
    )

    tagstore.upload_tags([tag], job.id)

    result2 = fetcher.download(vod_content, req)
    assert len(result2.successful_sources) == 1
    assert len(result2.failed) == 0

    req = DownloadRequest(
        stream_name="video",
        scope=VideoScope(
            start_time=0,
            end_time=60
        ),
        preserve_track="track2"
    )

    # shouldn't replace track2
    result3 = fetcher.download(vod_content, req)
    assert len(result3.successful_sources) == 2
    assert len(result3.failed) == 0

    # don't set preserve_track
    req = DownloadRequest(
        stream_name="video",
        scope=VideoScope(
            start_time=0,
            end_time=60
        ),
        preserve_track=""
    )

    result4 = fetcher.download(vod_content, req)
    assert len(result4.successful_sources) == 2
    assert len(result4.failed) == 0

    # audio instead
    req = DownloadRequest(
        stream_name="audio",
        scope=VideoScope(
            start_time=0,
            end_time=60
        ),
        preserve_track="track"
    )

    try:
        result5 = fetcher.download(vod_content, req)
        assert len(result5.successful_sources) == 2
        assert len(result5.failed) == 0
    except MissingResourceError:
        pass

def test_fetch_assets_with_preserve_track(
    fetcher: Fetcher, 
    assets_content: Content
):
    req1 = DownloadRequest(
        stream_name="assets",
        scope=AssetScope(
            assets=None
        ),
        preserve_track=""
    )
    
    result1 = fetcher.download(assets_content, req1)
    assert len(result1.successful_sources) > 0, "Should have downloaded some assets"
    assert len(result1.failed) == 0, "Should have no failed downloads initially"
    
    all_asset_names = [source.name for source in result1.successful_sources]
    selected_assets = all_asset_names[:3]
    
    req2 = DownloadRequest(
        stream_name="assets",
        scope=AssetScope(
            assets=selected_assets
        ),
        preserve_track=""
    )

    result2 = fetcher.download(assets_content, req2)
    assert len(result2.successful_sources) == len(selected_assets), "Should return all requested assets"
    assert len(result2.failed) == 0, "Should have no failed downloads"
    
    # Verify the returned assets match what we requested
    returned_asset_names = [source.name for source in result2.successful_sources]
    assert set(returned_asset_names) == set(selected_assets), "Returned assets should match requested assets"
    
    # Third test: Add tags for some assets and test preserve_track functionality
    tagstore = fetcher.ts
    
    job = tagstore.start_job(
        qhit=assets_content.qhit,
        stream="assets",
        author=fetcher.config.author,
        track="asset_track",
        q=assets_content
    )
    jobid = job.id
    
    # Tag the first two assets
    assets_to_tag = selected_assets[:2] if len(selected_assets) >= 2 else selected_assets
    tags = []
    for asset_name in assets_to_tag:
        tag = Tag(
            start_time=0,
            end_time=1,
            text="test asset tag",
            additional_info={},
            source=asset_name,
            jobid=jobid
        )
        tags.append(tag)
    
    tagstore.upload_tags(tags, jobid)
    
    req3 = DownloadRequest(
        stream_name="assets",
        scope=AssetScope(
            assets=selected_assets
        ),
        preserve_track="asset_track"
    )

    result3 = fetcher.download(assets_content, req3)
    returned_asset_names_after_tagging = [source.name for source in result3.successful_sources]
    
    for tagged_asset in assets_to_tag:
        assert tagged_asset not in returned_asset_names_after_tagging, f"Tagged asset {tagged_asset} should be excluded when preserve_track is set"
    
    # Should still return untagged assets
    untagged_assets = [asset for asset in selected_assets if asset not in assets_to_tag]
    for untagged_asset in untagged_assets:
        assert untagged_asset in returned_asset_names_after_tagging, f"Untagged asset {untagged_asset} should still be returned"
    
    # Fifth test: fetch assets with different preserve_track - should return all assets
    req4 = DownloadRequest(
        stream_name="assets",
        scope=AssetScope(
            assets=selected_assets
        ),
        preserve_track="different_track"
    )

    result4 = fetcher.download(assets_content, req4)
    returned_asset_names_different_track = [source.name for source in result4.successful_sources]
    assert set(returned_asset_names_different_track) == set(selected_assets), "Should return all assets when preserve_track doesn't match"
    
    # Upload tags to selected tags with author="user" and track="another track", check that downloading again
    # returns all the selected assets (tagger author is special)

    new_job = tagstore.start_job(
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
            additional_info={},
            source=asset,
            jobid=new_job.id
        )
        newtags.append(tag)
    tagstore.upload_tags(newtags, new_job.id)

    # Verify that the tags were uploaded correctly
    uploaded_tags = tagstore.find_tags(jobid=new_job.id)
    assert len(uploaded_tags) == len(newtags), "Not all tags were uploaded"
    for tag in newtags:
        assert tag in uploaded_tags, f"Tag {tag} was not found in uploaded tags"

    req5 = DownloadRequest(
        stream_name="assets",
        scope=AssetScope(
            assets=selected_assets
        ),
        preserve_track="asset_track"
    )

    result5 = fetcher.download(assets_content, req5)
    returned_asset_names_after_tagging = [source.name for source in result5.successful_sources]

    assert len(returned_asset_names_after_tagging) >= 1

    for assetname in returned_asset_names_after_tagging:
        assert assetname in selected_assets, f"Asset {assetname} should be in the selected assets after tagging"
        # tagged before
        assert assetname not in assets_to_tag
        assert assetname in untagged_assets

def test_find_default_audio_stream(
    fetcher: Fetcher,
    vod_content: Content,
):
    result = fetcher._find_default_audio_stream(vod_content)

    assert result == "stereo"