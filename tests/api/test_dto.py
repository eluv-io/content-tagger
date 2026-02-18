import pytest
from unittest.mock import Mock, MagicMock

from src.api.tagging.request_mapping import map_video_tag_dto
from src.fetch.model import AssetScope, LiveScope, TimeRangeScope, VideoScope
from src.tag_containers.registry import ContainerRegistry
from src.api.tagging.request_format import (
    StartJobsRequest, JobSpec, TaggerArgs, 
)
from src.tagging.fabric_tagging.model import TagArgs

@pytest.fixture
def mock_registry():
    """Mock ContainerRegistry for testing."""
    registry = Mock(spec=ContainerRegistry)
    # Mock model configs for different features
    def get_model_config(feature):
        if feature == "object_detection":
            return Mock(type="video")
        elif feature == "joe's processor":
            return Mock(type="processor")
        else:
            return Mock(type="audio")
    registry.get_model_config.side_effect = get_model_config
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

def test_vod_with_destination_qid(mock_registry, mock_content):
    """Test VOD mapping with destination_qid set."""
    args = StartJobsRequest(
        defaults=TaggerArgs(
            destination_qid="iq__destination",
            replace=True,
            scope={"type": "video", "start_time": 0, "end_time": 30}
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={"threshold": 0.5},
                overrides=None
            ),
            JobSpec(
                model="speech_recognition", 
                model_params={},
                overrides=TaggerArgs(scope={"stream": "audio"})
            )
        ]
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 2
    for tag_args in result:
        assert isinstance(tag_args, TagArgs)
        assert tag_args.destination_qid == "iq__destination"
        assert tag_args.replace == True

    assert isinstance(result[0].scope, VideoScope)
    assert isinstance(result[1].scope, VideoScope)
    assert result[0].scope.stream == "video"
    assert result[1].scope.stream == "audio"
    assert result[0].scope.start_time == 0
    assert result[0].scope.end_time == 30

def test_live_with_destination_qid(mock_registry, mock_content):
    """Test live mapping with destination_qid set."""
    args = StartJobsRequest(
        defaults=TaggerArgs(
            destination_qid="iq__destination",
            scope={"type": "livestream", "segment_length": 5, "max_duration": 60}
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={"threshold": 0.5},
                overrides=None
            )
        ]
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args, TagArgs)
    assert tag_args.destination_qid == "iq__destination"
    assert tag_args.replace == False  # Default value
    assert isinstance(tag_args.scope, LiveScope)
    assert tag_args.scope.chunk_size == 5
    assert tag_args.scope.max_duration == 60

def test_vod_without_destination_qid(mock_registry, mock_content):
    """Test VOD mapping without destination_qid (should default to empty string)."""
    args = StartJobsRequest(
        defaults=TaggerArgs(
            destination_qid="",
            replace=False
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={},
                overrides=None
            )
        ]
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

def test_live_without_destination_qid(mock_registry, mock_content):
    """Test live mapping without destination_qid (should default to empty string)."""
    args = StartJobsRequest(
        defaults=TaggerArgs(
            destination_qid="",
            scope={"type": "livestream"}
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={},
                overrides=None
            )
        ]
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

def test_asset_with_destination_qid(mock_registry, mock_content):
    """Test asset mapping with destination_qid set."""
    args = StartJobsRequest(
        defaults=TaggerArgs(
            destination_qid="iq__destination",
            replace=True,
            scope={"type": "assets", "assets": ["asset1.jpg", "asset2.png"]}
        ),
        jobs=[
            JobSpec(
                model="image_classification",
                model_params={"classes": ["cat", "dog"]},
                overrides=None
            ),
            JobSpec(
                model="object_detection",
                model_params={"threshold": 0.5},
                overrides=None
            )
        ]
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 2
    for tag_args in result:
        assert isinstance(tag_args, TagArgs)
        assert tag_args.destination_qid == "iq__destination"
        assert tag_args.replace == True
        assert isinstance(tag_args.scope, AssetScope)
        assert tag_args.scope.assets == ["asset1.jpg", "asset2.png"]

def test_asset_without_destination_qid(mock_registry, mock_content):
    """Test asset mapping without destination_qid (should default to empty string)."""
    args = StartJobsRequest(
        defaults=TaggerArgs(
            destination_qid="",
            scope={"type": "assets", "assets": ["asset1.jpg"]}
        ),
        jobs=[
            JobSpec(
                model="image_classification",
                model_params={},
                overrides=None
            )
        ]
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

def test_overrides_merge_with_defaults(mock_registry, mock_content):
    """Test that job overrides properly merge with defaults."""
    args = StartJobsRequest(
        defaults=TaggerArgs(
            destination_qid="iq__default",
            replace=False,
            scope={"type": "video", "start_time": 0, "end_time": 100}
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={},
                overrides=TaggerArgs(
                    destination_qid="iq__override",
                    scope={"start_time": 10, "end_time": 50}
                )
            )
        ]
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert tag_args.destination_qid == "iq__override"
    assert tag_args.replace == False  # From defaults
    assert isinstance(tag_args.scope, VideoScope)
    assert tag_args.scope.start_time == 10  # From override
    assert tag_args.scope.end_time == 50   # From override

def test_map_dto_with_joes_processor(mock_registry, mock_content):
    """Test mapping with a processor type model."""
    args = StartJobsRequest(
        defaults=TaggerArgs(),
        jobs=[
            JobSpec(
                model="joe's processor",
                model_params={},
                overrides=None
            )
        ]
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args.scope, TimeRangeScope)
    assert tag_args.scope.chunk_size == 600  # Default from ScopeProcessor

def test_asset_multiple_features_with_destination_qid(mock_registry, mock_content):
    """Test that all features get the same destination_qid."""
    args = StartJobsRequest(
        jobs=[
            JobSpec(
                model="feature1",
                model_params={},
                overrides=None
            ),
            JobSpec(
                model="feature2",
                model_params={},
                overrides=None
            ),
            JobSpec(
                model="feature3",
                model_params={},
                overrides=None
            )
        ],
        defaults=TaggerArgs(
            destination_qid="iq__shared_destination",
            scope={"type":"assets", "assets": ["asset.jpg"]},
        )
    )
    
    result = map_video_tag_dto(args, mock_registry, mock_content)
    
    assert len(result) == 3
    for tag_args in result:
        assert tag_args.destination_qid == "iq__shared_destination"