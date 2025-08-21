import pytest
import tempfile
import shutil
import os
from dotenv import load_dotenv
from typing import Generator
import time

from src.fetch.fetch_video import Fetcher
from src.fetch.types import FetcherConfig, VodDownloadRequest
from src.tags.tagstore import FilesystemTagStore, Tag, Job
from src.common.content import Content
from src.common.errors import MissingResourceError

# Load environment variables from .env file
load_dotenv()

VOD_QHIT = "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm"
LEGACY_VOD_QHIT = "hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU"

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def tag_store(temp_dir: str) -> FilesystemTagStore:
    """Create a FilesystemTagStore for testing"""
    return FilesystemTagStore(temp_dir)


@pytest.fixture
def fetcher_config(temp_dir: str) -> FetcherConfig:
    """Create a FetcherConfig for testing"""
    parts_path = os.path.join(temp_dir, "parts")
    os.makedirs(parts_path, exist_ok=True)
    
    return FetcherConfig(
        max_downloads=2,
        parts_path=parts_path
    )


@pytest.fixture
def fetcher(fetcher_config: FetcherConfig, tag_store: FilesystemTagStore) -> Fetcher:
    """Create a Fetcher instance for testing"""
    return Fetcher(config=fetcher_config, tagstore=tag_store)


@pytest.fixture
def legacy_vod_content() -> Content:
    auth_token = os.getenv("LEGACY_VOD_AUTH")
    
    if not auth_token:
        pytest.skip("LEGACY_VOD_AUTH not set in environment")
    
    try:
        return Content(qhit=LEGACY_VOD_QHIT, auth=auth_token)
    except Exception as e:
        pytest.skip(f"Failed to create legacy VOD content: {e}")


@pytest.fixture
def vod_content() -> Content:
    auth_token = os.getenv("VOD_AUTH")
    
    if not auth_token:
        pytest.skip("VOD_AUTH not set in environment")
    
    try:
        return Content(qhit=VOD_QHIT, auth=auth_token)
    except Exception as e:
        pytest.skip(f"Failed to create modern VOD content: {e}")

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
    req = VodDownloadRequest(
        stream_name="video",
        start_time=0,
        end_time=60,
        replace_track="track"
    )
    
    # Download once
    result1 = fetcher.download_stream(vod_content, req)
    assert len(result1.successful_sources) == 2
    assert len(result1.failed_part_hashes) == 0
    
    tagstore = fetcher.tagstore

    jobid = "abc"
    job = Job(
        id=jobid,
        qhit=vod_content.qhit,
        stream="video",
        timestamp=time.time(),
        author="tagger",
        track="track"
    )
    tagstore.start_job(job)

    first_source = result1.successful_sources[0].name

    tag = Tag(
        start_time=0,
        end_time=1,
        text="hello",
        additional_info={},
        source=first_source,
        jobid=jobid
    )

    tagstore.upload_tags([tag], jobid)

    result2 = fetcher.download_stream(vod_content, req)
    assert len(result2.successful_sources) == 1
    assert len(result2.failed_part_hashes) == 0

    req = VodDownloadRequest(
        stream_name="video",
        start_time=0,
        end_time=60,
        replace_track="track2"
    )

    # shouldn't replace track2
    result3 = fetcher.download_stream(vod_content, req)
    assert len(result3.successful_sources) == 2
    assert len(result3.failed_part_hashes) == 0

    # don't set replace_track
    req = VodDownloadRequest(
        stream_name="video",
        start_time=0,
        end_time=60,
        replace_track=""
    )

    result4 = fetcher.download_stream(vod_content, req)
    assert len(result4.successful_sources) == 2
    assert len(result4.failed_part_hashes) == 0

    # audio instead
    req = VodDownloadRequest(
        stream_name="stereo",
        start_time=0,
        end_time=60,
        replace_track="track"
    )

    try:
        result5 = fetcher.download_stream(vod_content, req)
        assert len(result5.successful_sources) == 2
        assert len(result5.failed_part_hashes) == 0
    except MissingResourceError:
        pass