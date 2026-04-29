
from unittest.mock import Mock
import pytest

from src.status.service import TaggingStatusService
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Batch
from src.tags.track_resolver import TrackResolver, TrackResolverConfig, TrackArgs


@pytest.fixture
def mock_tagstore():
    def _mock_tagstore(batches: list[Batch]) -> Tagstore:
        tagstore = Mock(spec=Tagstore)
        tagstore.find_batches.return_value = [b.id for b in batches]
        def get_batch_side_effect(batch_id: str, q=None):
            for b in batches:
                if b.id == batch_id:
                    return b
            return None
        tagstore.get_batch = get_batch_side_effect
        return tagstore
    return _mock_tagstore

@pytest.fixture
def track_resolver() -> TrackResolver:
    return TrackResolver(
        TrackResolverConfig(
            mapping={
                "llava": TrackArgs(name="llava_track", label="LLaVA"),
                "whisper": TrackArgs(name="whisper_track", label="Whisper"),
            }
        )
    )

@pytest.fixture
def get_status_service(mock_tagstore, track_resolver):
    def fn(batches: list[Batch]) -> TaggingStatusService:
        return TaggingStatusService(
            tagstore=mock_tagstore(batches),
            track_resolver=track_resolver,
        )
    return fn