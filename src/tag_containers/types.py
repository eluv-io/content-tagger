from dataclasses import dataclass
from typing_extensions import Literal

from src.common.resources import SystemResources
from src.tags.tagstore.types import Tag


@dataclass
class ModelConfig:
    name: str
    image: str
    type: Literal["video", "frame"]
    resources: SystemResources

@dataclass
class ContainerSpec:
    cachepath: str
    logspath: str
    tagspath: str
    fileargs: list[str]
    runconfig: dict
    model_config: ModelConfig 

@dataclass
class RegistryConfig:
    modconfigs: dict[str, ModelConfig]
    logspath: str
    tagspath: str
    cachepath: str

@dataclass
class ModelOutput:
    source_media: str
    tags: list[Tag]