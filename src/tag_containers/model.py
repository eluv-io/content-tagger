from dataclasses import dataclass
from typing_extensions import Literal

from src.tags.tagstore.types import Tag
from src.common.model import SystemResources

@dataclass
class ModelConfig:
    image: str
    type: Literal["audio", "video", "frame"]
    resources: SystemResources

@dataclass
class ContainerRequest:
    model_id: str
    file_args: list[str]
    run_config: dict
    # handle passed to the container for tracking
    job_id: str | None

    def __str__(self) -> str:
        return f"ContainerRequest(job_id={self.job_id}, model_id={self.model_id}, file_args={self.file_args}, run_config={self.run_config})"

@dataclass
class ContainerSpec:
    id: str
    cache_dir: str
    logs_path: str
    tags_dir: str
    file_args: list[str]
    run_config: dict
    model_config: ModelConfig

@dataclass
class RegistryConfig:
    # maps model_id -> ModelConfig
    model_configs: dict[str, ModelConfig]
    base_dir: str
    cache_dir: str

@dataclass
class ModelOutput:
    source_media: str
    tags: list[Tag]

@dataclass
class FrameTag:
    frame_idx: str
    confidence: float
    box: dict
    text: str