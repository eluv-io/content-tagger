

import os

import pytest

from src.common.model import ModelConfigStore
from src.api_extensions.models import list_models
from src.tag_containers.model import ModelConfig
from src.tags.track_resolver import TrackArgs, TrackResolver, LabelResolverConfig


@pytest.fixture
def listed_model_configs() -> dict[str, ModelConfig]:
    return {
        "test_model": ModelConfig(
            type="frame",
            description="Test model",
            category="Frame Level Detection",
            resources={"gpu": 1},
            image="localhost/test_model:latest",
            track_outputs=["test_model"]
        ),
        "test_model2": ModelConfig(
            type="processor",
            description="Test model 2",
            resources={"gpu": 1},
            image="localhost/test_model:latest",
            track_outputs=["another_model"],
            track_dependencies=["test_model"]
        ),
        # should be hidden from listing API due to empty description
        "hidden_model": ModelConfig(
            type="processor",
            description="",
            resources={"gpu": 1},
            image="localhost/test_model:latest",
            track_outputs=["test_model"],
            track_dependencies=["test_model"]
        )
    }


@pytest.fixture
def fake_resolver(model_configs):
    return TrackResolver(label_configs=LabelResolverConfig(
            mapping={"test_model": "TEST MODEL",
                    "another_model": "Some label"}
        ),
        model_configs=ModelConfigStore(model_configs)
    )

def test_listing(listed_model_configs):
    params_schema = {"type": "object", "properties": {"fps": {"type": "integer"}}}
    res = list_models(listed_model_configs, {"test_model": params_schema})
    models = res.models

    assert len(models) == 2
    assert models[0].name == "test_model"
    assert models[0].type == "frame"
    assert models[0].description == "Test model"
    assert models[0].tag_tracks[0].name == "test_model"
    assert models[0].dependencies == []
    assert models[0].category == "Frame Level Detection"
    assert models[1].name == "test_model2"
    assert models[1].type == "processor"
    assert models[1].description == "Test model 2"
    assert models[1].tag_tracks[0].name == "another_model"
    assert models[1].dependencies == ["test_model"]
    # empty category means unset
    assert models[1].category == ""
    assert models[0].params_schema == params_schema
    assert models[1].params_schema is None
