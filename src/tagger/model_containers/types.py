from dataclasses import dataclass
from typing_extensions import Literal

from src.tagger.resource_manager import SystemResources

@dataclass
class ContainerSpec:
    image: str
    cachepath: str
    logspath: str
    tagspath: str
    fileargs: list[str]
    runconfig: dict

@dataclass
class ModelConfig:
    name: str
    image: str
    type: Literal["video", "audio", "frame"]
    resources: SystemResources

@dataclass
class RegistryConfig:
    modconfigs: dict[str, ModelConfig]
    logspath: str
    tagspath: str
    cachepath: str