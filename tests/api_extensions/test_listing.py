

import os

import pytest

from src.api_extensions.models import list_models
from src.tag_containers.model import ModelConfig, RegistryConfig
from src.tag_containers.registry import ContainerRegistry
from src.tags.track_resolver import TrackArgs, TrackResolver, LabelResolverConfig


@pytest.fixture
def fake_registry(temp_dir):
    return ContainerRegistry(
        cfg=RegistryConfig(
            base_dir=temp_dir,
            cache_dir=temp_dir,
        ),
        model_configs={
            "test_model": ModelConfig(
                type="frame",
                description="Test model",
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
    )


@pytest.fixture
def fake_resolver(model_configs):
    return TrackResolver(label_configs=LabelResolverConfig(
            mapping={"test_model": "TEST MODEL",
                    "another_model": "Some label"}
        ),
        model_configs=model_configs 
    )

def test_listing(fake_registry):
    res = list_models(fake_registry.model_configs)
    models = res.models

    assert len(models) == 2
    assert models[0].name == "test_model"
    assert models[0].type == "frame"
    assert models[0].description == "Test model"
    assert models[0].tag_tracks[0].name == "test_model"
    assert models[0].dependencies == []
    assert models[1].name == "test_model2"
    assert models[1].type == "processor"
    assert models[1].description == "Test model 2"
    assert models[1].tag_tracks[0].name == "another_model"
    assert models[1].dependencies == ["test_model"]