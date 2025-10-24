import os
import shutil
import tempfile
import pytest
import dotenv
from unittest.mock import Mock

from src.api.tagging.dto_mapping import _is_live
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.tags.tagstore.rest_tagstore import RestTagstore
from src.tagging.fabric_tagging.model import FabricTaggerConfig
from src.tags.conversion import TagConverter, TagConverterConfig
from src.common.content import Content, ContentConfig, ContentFactory
from src.fetch.factory import FetchFactory
from src.fetch.model import FetcherConfig

dotenv.load_dotenv()

@pytest.fixture
def test_qid():
    return "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm"

@pytest.fixture
def live_q():
    token = os.getenv("LIVE_AUTH")
    if not token:
        pytest.skip("LIVE_AUTH not set in environment")
    cfg = ContentConfig(
        config_url="https://host-76-74-29-5.contentfabric.io/config?self&qspace=main", 
        parts_url="http://192.168.96.203/config?self&qspace=main",
        live_media_url="https://host-76-74-34-204.contentfabric.io/config?self&qspace=main"
    )
    factory = ContentFactory(cfg=cfg)
    q = factory.create_content(qhit="iq__HPzDaWpfmQzj2Afa3XFq2cpun5n", auth=token)
    if not _is_live(q):
        pytest.skip("livestream is not running")
    return q

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def media_dir(temp_dir: str) -> str:
    """Create a media directory for testing"""
    media_path = os.path.join(temp_dir, "media")
    os.makedirs(media_path, exist_ok=True)
    return media_path

@pytest.fixture
def tagger_config(media_dir) -> FabricTaggerConfig:
    return FabricTaggerConfig(media_dir=media_dir)

@pytest.fixture
def tag_converter() -> TagConverter:
    """Create a TagConverter with test configuration"""
    config = TagConverterConfig(
        interval=5,  # 5-minute buckets for testing
        name_mapping={
            "object": "Object Detection",
            "asr": "Speech to Text",
            "shot": "Shot Detection"
        },
        single_tag_tracks=[],
        coalesce_tracks=["asr"],
        max_sentence_words=10
    )
    return TagConverter(config)

@pytest.fixture
def qfactory() -> ContentFactory:
    cfg = ContentConfig(
        config_url="https://host-154-14-185-98.contentfabric.io/config?self&qspace=main", 
        parts_url="http://192.168.96.203/config?self&qspace=main",
        live_media_url="https://host-76-74-34-204.contentfabric.io/config?self&qspace=main"
    )
    factory = ContentFactory(cfg=cfg)
    return factory

@pytest.fixture
def writable_q(qfactory):
    """Check if TEST_AUTH and TEST_QWT are set in environment"""
    auth_token = os.getenv("TEST_AUTH")
    write_token = os.getenv("TEST_QWT")
    
    if not write_token or not auth_token:
        yield None
        return

    q = qfactory.create_content(qhit=write_token, auth=auth_token)

    yield q
    q.replace_metadata(metadata_subtree="video_tags", metadata={})
    
@pytest.fixture
def q(qfactory, test_qid):
    """Check if TEST_AUTH is set in environment"""
    auth_token = os.getenv("TEST_AUTH")
    
    if not auth_token:
        return Mock(qid=test_qid, qhit=test_qid)

    return qfactory.create_content(qhit=test_qid, auth=auth_token)

"""
@pytest.fixture
def q(writable_q, readonly_q, test_qid):
    #Create Content object with write token from environment
    auth_token = os.getenv("TEST_AUTH")
    write_token = os.getenv("TEST_QWT")

    if write_token:
        return writable_q
    elif auth_token:
        return readonly_q
    else:
        return Mock(qid=test_qid, qhit=test_qid)
"""

@pytest.fixture
def rest_tagstore(q: Content) -> RestTagstore:
    """Create a RestTagstore using TEST_TAGSTORE_HOST environment variable"""
    host = os.getenv("TEST_TAGSTORE_HOST") or ""
    ts = RestTagstore(base_url=host)

    if host:
        jobids = ts.find_jobs(q=q, limit=1000)
        for jobid in jobids:
            ts.delete_job(jobid, q=q)

    return ts

@pytest.fixture
def filesystem_tagstore(temp_dir: str) -> FilesystemTagStore:
    """Create a FilesystemTagStore with test data"""
    tagstore_dir = os.path.join(temp_dir, "tagstore")
    store = FilesystemTagStore(base_dir=tagstore_dir)
    return store

@pytest.fixture
def tag_store(rest_tagstore: RestTagstore, filesystem_tagstore: FilesystemTagStore):
    """Create appropriate tagstore based on TEST_TAGSTORE_HOST environment variable"""
    if os.getenv("TEST_TAGSTORE_HOST"):
        return rest_tagstore
    else:
        return filesystem_tagstore