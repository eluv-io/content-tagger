import os
import shutil
import tempfile
from unittest.mock import Mock
import pytest
import dotenv

from src.fetch.model import FetcherConfig, VideoScope
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.tags.tagstore.rest_tagstore import RestTagstore
from src.common.content import Content, ContentConfig, QAPIFactory
from src.tagging.fabric_tagging.queue.fs_jobstore import FsJobStore
from src.status.get_info import UserInfo, UserInfoResolver

dotenv.load_dotenv()

"""Fabric Integration Test Fixtures"""

@pytest.fixture
def content_config() -> ContentConfig:
    return ContentConfig(
        config_url="https://host-76-74-29-5.contentfabric.io/config?self&qspace=main", 
        parts_url="http://192.168.96.203/config?self&qspace=main",
        live_media_url="https://host-76-74-29-5.contentfabric.io/config?self&qspace=main"
    )

@pytest.fixture
def qfactory(content_config) -> QAPIFactory:
    factory = QAPIFactory(cfg=content_config)
    return factory

@pytest.fixture
def qid():
    return "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm"

@pytest.fixture
def q(qid):
    auth_token = os.getenv("TEST_AUTH") or ""

    return Content(qid=qid, token=auth_token)

@pytest.fixture
def qid_legacy():
    return "iq__cebzuQ8BqsWZyoUdnTXCe23fUgz"

@pytest.fixture
def q_legacy(qid_legacy):
    auth_token = os.getenv("TEST_AUTH")
    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")

    return Content(qid=qid_legacy, token=auth_token)

@pytest.fixture
def qid_live():
    return "iq__467CAS4BvPQ39go6aLmX6v3ZaTwD"

@pytest.fixture
def q_live(qid_live):
    token = os.getenv("LIVE_AUTH")
    if not token:
        pytest.skip("LIVE_AUTH not set in environment")

    return Content(qid=qid_live, token=token)

@pytest.fixture
def qid_assets():
    return "iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2"

@pytest.fixture
def q_assets(qid_assets):
    auth_token = os.getenv("ASSETS_AUTH")
    if not auth_token:
        pytest.skip("ASSETS_AUTH not set in environment")

    return Content(qid=qid_assets, token=auth_token)

"""Basic utility fixtures"""

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture()
def static_dir():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test-stuff')
    return test_dir

"""Tagstore Fixtures"""

@pytest.fixture
def rest_tagstore(q: Content) -> RestTagstore:
    """Create a RestTagstore using TEST_TAGSTORE_HOST environment variable"""
    host = os.getenv("TEST_TAGSTORE_HOST") or ""
    ts = RestTagstore(base_url=host, timeout=10)

    if host:
        jobids = ts.find_batches(q=q, limit=1000)
        for jobid in jobids:
            ts.delete_batch(jobid, q=q)

    return ts

@pytest.fixture
def filesystem_tagstore(temp_dir: str) -> FilesystemTagStore:
    """Create a FilesystemTagStore with test data"""
    tagstore_dir = os.path.join(temp_dir, "tagstore")
    store = FilesystemTagStore(base_dir=tagstore_dir)
    return store

@pytest.fixture
def tag_store(rest_tagstore: RestTagstore, filesystem_tagstore: FilesystemTagStore) -> Tagstore:
    """Create appropriate tagstore based on TEST_TAGSTORE_HOST environment variable"""
    if os.getenv("TEST_TAGSTORE_HOST"):
        return rest_tagstore
    else:
        return filesystem_tagstore
    
"""Fetcher Fixtures"""

@pytest.fixture
def fetcher_config() -> FetcherConfig:
    """Create a FetcherConfig for testing"""
    
    return FetcherConfig(
        author="tagger",
        max_downloads=4
    )

@pytest.fixture
def jobstore(temp_dir, fake_user_info_resolver) -> JobStore:
    """Create a JobStore for testing.
    
    If JOBSTORE_URL is set, a remote jobstore would be used — but that is not
    yet implemented.  If the variable is not set the local FsJobStore backed by
    a temporary directory is used instead.
    """
    url = os.getenv("JOBSTORE_URL")
    if url:
        raise NotImplementedError("Remote jobstore (JOBSTORE_URL) is not yet implemented")
    return FsJobStore(store_dir=os.path.join(temp_dir, "jobstore"), user_info_resolver=fake_user_info_resolver)

@pytest.fixture
def make_tag_args():
    """Factory for consistent TagArgs construction in tests."""
    def _make(
        feature: str = "caption",
        stream: str | None = None,
        destination_qid: str = "",
        # default to true for testing, but in real prod it's false
        replace: bool = True,
        run_config: dict | None = None,
        start_time: int = 0,
        end_time: int = 30,
        max_fetch_retries: int = 3
    ) -> TagArgs:
        if stream is None:
            stream = "audio" if feature == "asr" else "video"
        return TagArgs(
            feature=feature,
            run_config=run_config or {},
            scope=VideoScope(stream=stream, start_time=start_time, end_time=end_time),
            replace=replace,
            destination_qid=destination_qid,
            max_fetch_retries=max_fetch_retries
        )
    return _make

@pytest.fixture
def fake_user_info_resolver():
    return Mock(
        get_user_info=Mock(return_value=UserInfo(
            user_adr="0x123",
            is_tenant_admin=True,
                is_content_admin=True
            )),
        get_tenant=Mock(return_value="tenant1")
    )