
from unittest.mock import Mock
import pytest

from src.common.model import ModelConfig
from src.status.service import TaggingStatusService
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Batch
from src.tags.track_resolver import TrackResolver, LabelResolverConfig, TrackArgs


@pytest.fixture
def mock_tagstore():
    def _mock_tagstore(batches: list[Batch]) -> Tagstore:
        tagstore = Mock(spec=Tagstore)
        tagstore.find_batches.return_value = batches
        def get_batch_side_effect(batch_id: str, q=None):
            for b in batches:
                if b.id == batch_id:
                    return b
            return None
        tagstore.get_batch = get_batch_side_effect
        return tagstore
    return _mock_tagstore

@pytest.fixture
def model_configs():
    return {
        "llava": Mock(
            type="frame",
            description="Test model",
            track_outputs=["llava_track"]
        ),
        "whisper": Mock(
            type="audio",
            description="Test model 2",
            track_outputs=["whisper_track"]
        ),
    }

@pytest.fixture
def track_resolver(model_configs) -> TrackResolver:
    return TrackResolver(
        label_configs=LabelResolverConfig(
            mapping={
                "llava_track": "LLAVA",
                "whisper_track": "Whisper",
            }
        ),
        model_configs=model_configs
    )

@pytest.fixture
def get_status_service(mock_tagstore, track_resolver):
    def fn(batches: list[Batch]) -> TaggingStatusService:
        return TaggingStatusService(
            tagstore=mock_tagstore(batches),
            track_resolver=track_resolver,
        )
    return fn