from dataclasses import dataclass
from typing_extensions import Literal

from src.common.resources import SystemResources
from src.tags.tagstore.types import Tag


@dataclass
class ModelConfig:
    image: str
    type: Literal["audio", "video", "frame"]
    resources: SystemResources

@dataclass
class ContainerRequest:
    model: str
    file_args: list[str]
    run_config: dict
    # helps to configure where logs/tags go in filesystem
    job_id: str

@dataclass
class ContainerSpec:
    cache_dir: str
    logs_dir: str
    tags_dir: str
    file_args: list[str]
    run_config: dict
    model_config: ModelConfig

@dataclass
class RegistryConfig:
    model_configs: dict[str, ModelConfig]
    base_dir: str
    cache_dir: str

@dataclass
class ModelOutput:
    source_media: str
    tags: list[Tag]