import os
import shutil
import tempfile
import pytest
import dotenv
from unittest.mock import Mock

from src.api.tagging.request_mapping import is_live
from src.fetch.model import FetcherConfig
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.tags.tagstore.rest_tagstore import RestTagstore
from src.common.content import Content, ContentConfig, ContentFactory

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
def qfactory(content_config) -> ContentFactory:
    factory = ContentFactory(cfg=content_config)
    return factory

@pytest.fixture
def qid():
    return "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm"

@pytest.fixture
def q(qfactory, qid):
    auth_token = os.getenv("TEST_AUTH")
    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")

    return qfactory.create_content(qhit=qid, auth=auth_token)

@pytest.fixture
def qid_legacy():
    return "iq__cebzuQ8BqsWZyoUdnTXCe23fUgz"

@pytest.fixture
def q_legacy(qfactory, qid_legacy):
    auth_token = os.getenv("TEST_AUTH")
    if not auth_token:
        pytest.skip("TEST_AUTH not set in environment")

    return qfactory.create_content(qhit=qid_legacy, auth=auth_token)

@pytest.fixture
def qid_live():
    return "iq__467CAS4BvPQ39go6aLmX6v3ZaTwD"

@pytest.fixture
def q_live(qfactory, qid_live):
    token = os.getenv("LIVE_AUTH")
    if not token:
        pytest.skip("LIVE_AUTH not set in environment")

    q = qfactory.create_content(qhit=qid_live, auth=token)

    if not is_live(q):
        pytest.skip("livestream is not running")

    return q

@pytest.fixture
def qid_assets():
    return "iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2"

@pytest.fixture
def q_assets(qfactory, qid_assets):
    auth_token = os.getenv("ASSETS_AUTH")
    if not auth_token:
        pytest.skip("ASSETS_AUTH not set in environment")

    return qfactory.create_content(qhit=qid_assets, auth=auth_token)


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