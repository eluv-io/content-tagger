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
class ContainerSpec:
    cache_path: str
    logs_path: str
    tags_path: str
    file_args: list[str]
    run_config: dict
    model_config: ModelConfig

@dataclass
class RegistryConfig:
    model_configs: dict[str, ModelConfig]
    logs_path: str
    tags_path: str
    cache_path: str

@dataclass
class ModelOutput:
    source_media: str
    tags: list[Tag]