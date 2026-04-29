
import pytest
from unittest.mock import Mock

from src.tag_containers.model import ModelTag
from src.tagging.fabric_tagging.model import TagContentStatusReport, UploadStatus
from src.tagging.fabric_tagging.source_resolver import SourceResolver
from src.tagging.uploading.uploader import UploadSession
from src.tags.tagstore import filesystem_tagstore
from src.tags.tagstore.abstract import Tagstore

@pytest.fixture
def source_resolver(tag_store, track_resolver):
    return SourceResolver(
        tagstore=tag_store,
        track_resolver=track_resolver
    )

def test_source_resolver(q, source_resolver):
    tagstore = source_resolver.tagstore
    track = source_resolver.track_resolver.resolve("asr")
    batch = tagstore.create_batch(
        q=q,
        author="tagger",
        track=track.name,
    )

    tagstore.update_batch(
        batch_id=batch.id,
        additional_info={
            "tagger": {
                "upload_status": {
                    "uploaded_sources": ["source1", "source2"]
                }
            }
        },
        q=q
    )

    assert source_resolver.resolve(q, model="asr") == ["source1", "source2"]