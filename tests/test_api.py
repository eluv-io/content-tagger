import pytest
from unittest.mock import MagicMock, Mock
from src.api.tagging.dto_mapping import map_video_tag_dto, map_asset_tag_dto
from src.api.tagging.format import TagAPIArgs, LiveTagAPIArgs, ImageTagAPIArgs, ModelParams
from src.tagging.fabric_tagging.model import TagArgs
from src.fetch.model import VideoScope, LiveScope, AssetScope
from src.common.content import Content
from src.tag_containers.registry import ContainerRegistry

@pytest.fixture
def mock_registry():
    """Mock ContainerRegistry for testing."""
    registry = Mock(spec=ContainerRegistry)
    # Mock model configs for different features
    registry.get_model_config.side_effect = lambda feature: Mock(
        type="video" if feature == "object_detection" else "audio"
    )
    return registry

@pytest.fixture
def mock_content():
    """Mock Content object for testing."""
    content = MagicMock()
    # Mock metadata for audio stream lookup
    content.content_object_metadata.return_value = {
        "audio_en_2ch": {
            "codec_type": "audio",
            "language": "en",
            "channels": 2
        }
    }
    content.qhit = "iq__source"
    return content

# Video Tag DTO Tests

def test_vod_with_destination_qid(mock_registry, mock_content):
    """Test VOD mapping with destination_qid set."""
    args = TagAPIArgs(
        features={
            "object_detection": ModelParams(model={"threshold": 0.5}),
            "speech_recognition": ModelParams(stream="audio", model={})
        },
        destination_qid="iq__destination",
        replace=True,
        start_time=0,
        end_time=30
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 2
    for tag_args in result:
        assert isinstance(tag_args, TagArgs)
        assert tag_args.destination_qid == "iq__destination"
        assert tag_args.replace == True
    
    # Check specific scopes
    assert isinstance(result[0].scope, VideoScope)
    assert isinstance(result[1].scope, VideoScope)
    assert result[0].scope.stream == "video"  # object_detection defaults to video
    assert result[1].scope.stream == "audio"  # speech_recognition uses specified stream

def test_live_with_destination_qid(mock_registry, mock_content):
    """Test live mapping with destination_qid set."""
    args = LiveTagAPIArgs(
        features={
            "object_detection": ModelParams(model={"threshold": 0.5})
        },
        destination_qid="iq__destination",
        segment_length=5,
        max_duration=60
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args, TagArgs)
    assert tag_args.destination_qid == "iq__destination"
    assert tag_args.replace == False  # Live defaults to False
    assert isinstance(tag_args.scope, LiveScope)
    assert tag_args.scope.chunk_size == 5
    assert tag_args.scope.max_duration == 60

def test_vod_without_destination_qid(mock_registry, mock_content):
    """Test VOD mapping without destination_qid (should default to empty string)."""
    args = TagAPIArgs(
        features={
            "object_detection": ModelParams(model={})
        },
        destination_qid="",  # Explicitly empty
        replace=False
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

def test_live_without_destination_qid(mock_registry, mock_content):
    """Test live mapping without destination_qid (should default to empty string)."""
    args = LiveTagAPIArgs(
        features={
            "object_detection": ModelParams(model={})
        },
        destination_qid=""  # Explicitly empty
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

# Asset Tag DTO Tests

def test_asset_with_destination_qid():
    """Test asset mapping with destination_qid set."""
    args = ImageTagAPIArgs(
        features={
            "image_classification": ModelParams(model={"classes": ["cat", "dog"]}),
            "object_detection": ModelParams(model={"threshold": 0.5})
        },
        destination_qid="iq__destination",
        replace=True,
        assets=["asset1.jpg", "asset2.png"]
    )
    
    result = map_asset_tag_dto(args)
    
    assert len(result) == 2
    for tag_args in result:
        assert isinstance(tag_args, TagArgs)
        assert tag_args.destination_qid == "iq__destination"
        assert tag_args.replace == True
        assert isinstance(tag_args.scope, AssetScope)
        assert tag_args.scope.assets == ["asset1.jpg", "asset2.png"]

def test_asset_without_destination_qid():
    """Test asset mapping without destination_qid (should default to empty string)."""
    args = ImageTagAPIArgs(
        features={
            "image_classification": ModelParams(model={})
        },
        destination_qid="",  # Explicitly empty
        assets=["asset1.jpg"]
    )
    
    result = map_asset_tag_dto(args)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

def test_asset_multiple_features_with_destination_qid():
    """Test that all features get the same destination_qid."""
    args = ImageTagAPIArgs(
        features={
            "feature1": ModelParams(model={}),
            "feature2": ModelParams(model={}),
            "feature3": ModelParams(model={})
        },
        destination_qid="iq__shared_destination",
        assets=["asset.jpg"]
    )
    
    result = map_asset_tag_dto(args)
    
    assert len(result) == 3
    for tag_args in result:
        assert tag_args.destination_qid == "iq__shared_destination"