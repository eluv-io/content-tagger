from unittest import result

import pytest
from unittest.mock import Mock, patch
from flask import Flask

from src.api.arg_resolver import ArgsResolver
from src.api.arg_resolver import ArgsResolver
from src.fetch.model import AssetScope, LiveScope, TimeRangeScope, VideoScope
from src.tag_containers.registry import ContainerRegistry
from src.api.tagging.request_format import (
    StartJobsRequest, JobSpec, TaggerOptions, StatusRequest,
)
from src.common.errors import BadRequestError
from src.tagging.fabric_tagging.model import TagArgs
from src.api.tagging.handlers import _parse_status_request

@pytest.fixture
def mock_registry():
    """Mock ContainerRegistry for testing."""
    registry = Mock(spec=ContainerRegistry)
    # Mock model configs for different features
    def get_model_config(feature):
        if feature == "object_detection":
            return Mock(type="video")
        elif feature == "image_classification":
            return Mock(type="frame")
        elif feature == "joe's processor":
            return Mock(type="processor")
        elif feature == "audio_classification":
            return Mock(type="audio")
        else:
            return Mock(type="video")
    registry.get_model_config.side_effect = get_model_config
    return registry

@pytest.fixture
def mock_content():
    """Mock Content object for testing."""
    content = Mock(qid="iq__source", token="")
    return content

@pytest.fixture
def mock_qfactory():
    qfactory = Mock()
    return qfactory

@pytest.fixture
def resolver(mock_registry, mock_qfactory):
    """Import the resolver function for testing."""
    return ArgsResolver(mock_registry, mock_qfactory)

def test_auto_detect_livestream(resolver, mock_content):
    """Test that livestream scope is auto-detected when segment_length is provided."""
    args = StartJobsRequest(
        options=TaggerOptions(),
        jobs=[
            JobSpec(
                model="object_detection",
            )
        ]
    )

    resolver.is_live_content = Mock(return_value=True)
    resolver.find_default_audio_stream = Mock(return_value="audio")
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args.scope, LiveScope)
    assert tag_args.scope.chunk_size > 0

def test_resolve_audio_stream(resolver, mock_content):
    """Test that audio stream is correctly resolved for audio models."""
    args = StartJobsRequest(
        options=TaggerOptions(),
        jobs=[
            JobSpec(
                model="audio_classification",
            )
        ]
    )

    resolver.is_live_content = Mock(return_value=False)
    resolver.find_default_audio_stream = Mock(return_value="audio")
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args.scope, VideoScope)
    assert tag_args.scope.stream == "audio"

def test_detect_processor_scope(resolver, mock_content):
    """Test that processor scope is used for processor type models."""
    args = StartJobsRequest(
        options=TaggerOptions(),
        jobs=[
            JobSpec(
                model="joe's processor",
            )
        ]
    )

    resolver.is_live_content = Mock(return_value=False)
    resolver.find_default_audio_stream = Mock(return_value="audio")
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args.scope, TimeRangeScope)
    assert tag_args.scope.chunk_size > 0

def test_vod_with_destination_qid(resolver, mock_content):
    """Test VOD mapping with destination_qid set."""
    args = StartJobsRequest(
        options=TaggerOptions(
            destination_qid="iq__destination",
            replace=True,
            scope={"type": "video", "start_time": 0, "end_time": 30}
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={"threshold": 0.5},
                overrides=TaggerOptions()
            ),
            JobSpec(
                model="speech_recognition", 
                model_params={},
                overrides=TaggerOptions(scope={"stream": "audio"})
            )
        ]
    )
    
    resolver.is_live_content = Mock(return_value=False)
    resolver.find_default_audio_stream = Mock(return_value="audio")
    result = resolver.resolve(args, mock_content)
    
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

def test_live_with_destination_qid(resolver, mock_content):
    """Test live mapping with destination_qid set."""
    args = StartJobsRequest(
        options=TaggerOptions(
            destination_qid="iq__destination",
            scope={"type": "livestream", "segment_length": 5, "max_duration": 60}
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={"threshold": 0.5},
                overrides=TaggerOptions()
            )
        ]
    )

    resolver.is_live_content = Mock(return_value=True)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args, TagArgs)
    assert tag_args.destination_qid == "iq__destination"
    assert tag_args.replace == False
    assert isinstance(tag_args.scope, LiveScope)
    assert tag_args.scope.chunk_size == 5
    assert tag_args.scope.max_duration == 60

def test_vod_without_destination_qid(resolver, mock_content):
    """Test VOD mapping without destination_qid (should default to empty string)."""
    args = StartJobsRequest(
        options=TaggerOptions(
            destination_qid="",
            replace=False
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={},
                overrides=TaggerOptions()
            )
        ]
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

def test_live_without_destination_qid(resolver, mock_content):
    """Test live mapping without destination_qid (should default to empty string)."""
    args = StartJobsRequest(
        options=TaggerOptions(
            destination_qid="",
            scope={"type": "livestream"}
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={},
                overrides=TaggerOptions()
            )
        ]
    )
    
    resolver.is_live_content = Mock(return_value=True)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

def test_asset_with_destination_qid(resolver, mock_content):
    """Test asset mapping with destination_qid set."""
    args = StartJobsRequest(
        options=TaggerOptions(
            destination_qid="iq__destination",
            replace=True,
            scope={"type": "assets", "assets": ["asset1.jpg", "asset2.png"]}
        ),
        jobs=[
            JobSpec(
                model="image_classification",
                model_params={"classes": ["cat", "dog"]},
                overrides=TaggerOptions()
            ),
            JobSpec(
                model="object_detection",
                model_params={"threshold": 0.5},
                overrides=TaggerOptions()
            )
        ]
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 2
    for tag_args in result:
        assert isinstance(tag_args, TagArgs)
        assert tag_args.destination_qid == "iq__destination"
        assert tag_args.replace == True
        assert isinstance(tag_args.scope, AssetScope)
        assert tag_args.scope.assets == ["asset1.jpg", "asset2.png"]

def test_asset_without_destination_qid(resolver, mock_content):
    """Test asset mapping without destination_qid (should default to empty string)."""
    args = StartJobsRequest(
        options=TaggerOptions(
            destination_qid="",
            scope={"type": "assets", "assets": ["asset1.jpg"]}
        ),
        jobs=[
            JobSpec(
                model="image_classification",
                model_params={},
                overrides=TaggerOptions()
            )
        ]
    )
    resolver.is_live_content = Mock(return_value=False)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    assert result[0].destination_qid == ""

def test_overrides_merge_with_defaults(resolver, mock_content):
    """Test that job overrides properly merge with defaults."""
    args = StartJobsRequest(
        options=TaggerOptions(
            destination_qid="iq__default",
            replace=False,
            scope={"type": "video", "start_time": 0, "end_time": 100}
        ),
        jobs=[
            JobSpec(
                model="object_detection",
                model_params={},
                overrides=TaggerOptions(
                    destination_qid="iq__override",
                    scope={"start_time": 10, "end_time": 50}
                )
            )
        ]
    )
    resolver.is_live_content = Mock(return_value=False)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert tag_args.destination_qid == "iq__override"
    assert tag_args.replace == False  # From defaults
    assert isinstance(tag_args.scope, VideoScope)
    assert tag_args.scope.start_time == 10  # From override
    assert tag_args.scope.end_time == 50   # From override

def test_map_dto_with_joes_processor(resolver, mock_content):
    """Test mapping with a processor type model."""
    args = StartJobsRequest(
        options=TaggerOptions(),
        jobs=[
            JobSpec(
                model="joe's processor",
                model_params={},
                overrides=TaggerOptions()
            )
        ]
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args.scope, TimeRangeScope)
    assert tag_args.scope.chunk_size == 600  # Default from ScopeProcessor

def test_asset_multiple_features_with_destination_qid(resolver, mock_content):
    """Test that all features get the same destination_qid."""
    args = StartJobsRequest(
        jobs=[
            JobSpec(
                model="feature1",
                model_params={},
                overrides=TaggerOptions()
            ),
            JobSpec(
                model="feature2",
                model_params={},
                overrides=TaggerOptions()
            ),
            JobSpec(
                model="feature3",
                model_params={},
                overrides=TaggerOptions()
            )
        ],
        options=TaggerOptions(
            destination_qid="iq__shared_destination",
            scope={"type":"assets", "assets": ["asset.jpg"]},
        )
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 3
    for tag_args in result:
        assert tag_args.destination_qid == "iq__shared_destination"

def test_just_model_name(resolver, mock_content):
    """Test that just providing a model name works with all defaults."""
    args = StartJobsRequest(
        jobs=[
            JobSpec(
                model="object_detection",
            )
        ],
        options=TaggerOptions()
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver.resolve(args, mock_content)
    
    assert len(result) == 1
    tag_args = result[0]
    assert tag_args.feature == "object_detection"
    assert tag_args.destination_qid == ""
    assert tag_args.replace == False
    assert isinstance(tag_args.scope, VideoScope)

def test_audio_stream_mapping_live(q_live, resolver):
    args = StartJobsRequest(
        jobs=[
            JobSpec(
                model="object_detection",
                overrides=TaggerOptions(scope={"stream": "audio"})
            )
        ],
        options=TaggerOptions()
    )
    resolver.is_live_content = Mock(return_value=True)
    result = resolver.resolve(args, q_live)
    assert len(result) == 1
    tag_args = result[0]
    assert isinstance(tag_args.scope, LiveScope)
    assert tag_args.scope.stream == "audio"

def test_set_defaults_basic(resolver, mock_content):
    """Test basic functionality with minimal parameters."""
    defaults = TaggerOptions()
    job = JobSpec(model="object_detection")
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert isinstance(result, TagArgs)
    assert result.feature == "object_detection"
    assert result.destination_qid == ""
    assert result.replace == False
    assert result.max_fetch_retries == 3
    assert isinstance(result.scope, VideoScope)
    assert result.scope.end_time > 0
    assert result.scope.stream == "video"

def test_set_defaults_with_all_defaults(resolver, mock_content):
    """Test with all default values set."""
    defaults = TaggerOptions(
        destination_qid="default_qid",
        replace=True,
        max_fetch_retries=5,
        scope={"start_time": 10, "end_time": 100}
    )
    job = JobSpec(
        model="object_detection",
        model_params={"threshold": 0.8}
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert result.feature == "object_detection"
    assert result.run_config == {"threshold": 0.8}
    assert result.destination_qid == "default_qid"
    assert result.replace == True
    assert result.max_fetch_retries == 5
    assert isinstance(result.scope, VideoScope)
    assert result.scope.start_time == 10
    assert result.scope.end_time == 100

def test_set_defaults_job_overrides_precedence(resolver, mock_content):
    """Test that job overrides take precedence over defaults."""
    defaults = TaggerOptions(
        destination_qid="default_qid",
        replace=True,
        max_fetch_retries=5,
        scope={"start_time": 10, "end_time": 100}
    )
    job = JobSpec(
        model="object_detection",
        overrides=TaggerOptions(
            destination_qid="override_qid",
            replace=False,
            max_fetch_retries=7,
            scope={"start_time": 20, "end_time": 200}
        )
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert result.destination_qid == "override_qid"
    assert result.replace == False
    assert result.max_fetch_retries == 7
    assert isinstance(result.scope, VideoScope)
    assert result.scope.start_time == 20
    assert result.scope.end_time == 200

def test_set_defaults_partial_overrides(resolver, mock_content):
    """Test partial overrides - some from defaults, some from overrides."""
    defaults = TaggerOptions(
        destination_qid="default_qid",
        replace=True,
        max_fetch_retries=5
    )
    job = JobSpec(
        model="object_detection",
        overrides=TaggerOptions(
            destination_qid="override_qid",
            # replace and max_fetch_retries not overridden
        )
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert result.destination_qid == "override_qid"  # From override
    assert result.replace == True  # From defaults
    assert result.max_fetch_retries == 5  # From defaults

def test_set_defaults_processor_model(resolver, mock_content):
    """Test with processor type model."""
    defaults = TaggerOptions()
    job = JobSpec(model="joe's processor")
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert isinstance(result.scope, TimeRangeScope)
    assert result.scope.chunk_size > 0

def test_set_defaults_audio_model(resolver, mock_content):
    """Test with audio type model."""
    defaults = TaggerOptions()
    job = JobSpec(model="audio_classification")
    
    resolver.is_live_content = Mock(return_value=False)
    resolver.find_default_audio_stream = Mock(return_value="audio_stream")
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert isinstance(result.scope, VideoScope)
    assert result.scope.stream == "audio_stream"
    assert result.scope.end_time > 0

def test_set_defaults_live_content(resolver, mock_content):
    """Test with live content."""
    defaults = TaggerOptions()
    job = JobSpec(model="object_detection")
    
    resolver.is_live_content = Mock(return_value=True)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert isinstance(result.scope, LiveScope)
    assert result.scope.chunk_size > 0
    assert result.scope.stream == "video"

def test_set_defaults_live_content_audio_model(resolver, mock_content):
    """Test with live content."""
    defaults = TaggerOptions()
    job = JobSpec(model="audio_classification", overrides=TaggerOptions(scope={"stream": "audio_stream"}))
    
    resolver.is_live_content = Mock(return_value=True)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert isinstance(result.scope, LiveScope)
    assert result.scope.chunk_size > 0
    assert result.scope.stream == "audio_stream"

def test_set_defaults_asset_scope(resolver, mock_content):
    """Test with asset scope configuration."""
    defaults = TaggerOptions(
        scope={"type": "assets", "assets": ["image1.jpg", "image2.png"]}
    )
    job = JobSpec(model="object_detection")
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert isinstance(result.scope, AssetScope)
    assert result.scope.assets == ["image1.jpg", "image2.png"]

def test_set_defaults_scope_merge_priority(resolver, mock_content):
    """Test scope merging priority: model defaults < request defaults < job overrides."""
    defaults = TaggerOptions(
        scope={"start_time": 5, "end_time": 50}  # Request level
    )
    job = JobSpec(
        model="object_detection",
        overrides=TaggerOptions(
            scope={"end_time": 200}  # Job level override
        )
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert isinstance(result.scope, VideoScope)
    assert result.scope.start_time == 5
    assert result.scope.end_time == 200
    assert result.scope.stream == "video"

def test_set_defaults_invalid_scope_raises_error(resolver, mock_content):
    """Test that invalid scope configuration raises BadRequestError."""
    defaults = TaggerOptions(
        scope={"type": "invalid_type"}
    )
    job = JobSpec(model="object_detection")
    
    resolver.is_live_content = Mock(return_value=False)
    with pytest.raises(BadRequestError):
        resolver._set_defaults(mock_content, defaults, job)

def test_set_defaults_processor_live_content_error(resolver, mock_content):
    """Test that processor model with live content raises error."""
    defaults = TaggerOptions()
    job = JobSpec(model="joe's processor")
    
    resolver.is_live_content = Mock(return_value=True)
    with pytest.raises(BadRequestError):
        resolver._set_defaults(mock_content, defaults, job)

def test_set_defaults_empty_overrides(resolver, mock_content):
    """Test with empty overrides object."""
    defaults = TaggerOptions(
        destination_qid="test_qid",
        replace=True
    )
    job = JobSpec(
        model="object_detection",
        overrides=TaggerOptions()  # Empty overrides
    )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    # Should use defaults since overrides are None/empty
    assert result.destination_qid == "test_qid"
    assert result.replace == True

def test_set_defaults_complex_scope_override(resolver, mock_content):
    """Test complex scope override scenarios."""
    defaults = TaggerOptions(
        scope={
            "type": "video",
            "start_time": 0,
            "end_time": 100,
            "stream": "default_stream"
        }
    )
    job = JobSpec(
        model="object_detection",
        overrides=TaggerOptions(
            scope={
                "end_time": 500,
                "stream": "override_stream"
            }
        )
   )
    
    resolver.is_live_content = Mock(return_value=False)
    result = resolver._set_defaults(mock_content, defaults, job)
    
    assert isinstance(result.scope, VideoScope)
    assert result.scope.start_time == 0  # From defaults
    assert result.scope.end_time == 500  # From override
    assert result.scope.stream == "override_stream"  # From override


def test_parse_status_request():
    app = Flask(__name__)

    def parse(qs=""):
        with app.test_request_context(f"/status{qs}"):
            return _parse_status_request()

    # Defaults
    assert parse() == StatusRequest()

    # Int coercion from query string
    req = parse("?start=10&limit=5")
    assert req.start == 10 and req.limit == 5

    # All fields together
    req = parse("?start=2&limit=20&status=done&tenant=fox&user=bob")
    assert req == StatusRequest(start=2, limit=20, status="done", tenant="fox", user="bob")

    # 'authorization' param is silently stripped
    assert parse("?limit=3&authorization=secret_token").limit == 3

    # Unknown field raises (strict mode)
    with pytest.raises(BadRequestError):
        parse("?bogus_field=xyz")

    # Non-numeric value for int field raises
    with pytest.raises(BadRequestError):
        parse("?limit=abc")

    req = parse("?start=2&status=done&tenant=fox&user=bob")
    assert req == StatusRequest(start=2, limit=None, status="done", tenant="fox", user="bob")