

import os

import pytest

from src.api_extensions.models import list_models
from src.tag_containers.model import ModelConfig, RegistryConfig
from src.tag_containers.registry import ContainerRegistry
from src.tags.track_resolver import TrackArgs, TrackResolver, TrackResolverConfig


@pytest.fixture
def fake_registry(temp_dir):
    return ContainerRegistry(
        cfg=RegistryConfig(
            base_dir=temp_dir,
            cache_dir=temp_dir,
            model_configs={
                "test_model": ModelConfig(
                    type="frame",
                    resources={"gpu": 1},
                    image="localhost/test_model:latest"
                ),
                "test_model2": ModelConfig(
                    type="processor",
                    resources={"gpu": 1},
                    image="localhost/test_model:latest"
                )
            }
        )
    )


@pytest.fixture
def fake_resolver():
    return TrackResolver(cfg=TrackResolverConfig(
        mapping={"test_model": TrackArgs(name="test_model", label="TEST MODEL"),
                 "test_model2": TrackArgs(name="another_model", label="Some label")}
    ))

def test_listing(fake_registry, fake_resolver):
    res = list_models(fake_registry, fake_resolver)
    models = res.models

    assert len(models) == 2
    assert models[0].name == "test_model"
    assert models[0].type == "frame"
    assert models[0].tag_tracks[0].name == "test_model"
    assert models[0].tag_tracks[0].label == "TEST MODEL"

    assert models[1].name == "test_model2"
    assert models[1].type == "processor"
    assert models[1].tag_tracks[0].name == "another_model"
    assert models[1].tag_tracks[0].label == "Some label"