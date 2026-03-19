from dataclasses import dataclass, field
from typing_extensions import Literal

from src.tags.tagstore.model import Tag
from src.common.model import SystemResources

MediaInput = list[str] | str

@dataclass(frozen=True)
class ContainerInfo:
    image_name: str
    annotations: dict

@dataclass
class ModelConfig:
    """
    Describes static attributes of a model
    """
    image: str
    type: Literal["audio", "video", "frame", "processor"]
    resources: SystemResources

@dataclass
class ContainerSpec:
    """
    Contains all information needed to run a container including static attributes
    and runtime parameters.
    """

    # handle 
    id: str
    # volume to cache stuff in, i.e model weights
    cache_dir: str
    logs_path: str
    # path to the JSONL output file
    output_path: str
    # media files to run the model on
    media_input: MediaInput
    # runtime params passed to the model (unique schema per model)
    run_config: dict
    # static attributes of the container to run
    model_config: ModelConfig

@dataclass
class ContainerRequest:
    # e.g: "caption", "asr" (same model handle used in the API)
    model_id: str
    # Either a list of files or a single directory
    media_input: MediaInput
    # runtime params passed to the model
    run_config: dict
    # handle passed to the container for tracking
    job_id: str | None

    def __str__(self) -> str:
        return f"ContainerRequest(job_id={self.job_id}, model_id={self.model_id}, media_input={self.media_input}, run_config={self.run_config})"

@dataclass
class RegistryConfig:
    # maps model_id -> ModelConfig
    model_configs: dict[str, ModelConfig]
    base_dir: str
    cache_dir: str

@dataclass(frozen=True)
class Progress:
    source_media: str

@dataclass(frozen=True)
class Error:
    message: str
    source_media: str | None = None

@dataclass(frozen=True)
class ModelTag:
    """
    Represents a tag produced at the model level
    """
    start_time: int
    end_time: int
    text: str
    source_media: str
    model_track: str
    frame_info: dict | None = None
    additional_info: dict | None = None

    def __hash__(self) -> int:
        fi_hash = self.frame_info.get("frame_idx") if self.frame_info else None
        return hash((self.start_time, self.end_time, self.text, self.source_media, self.model_track, fi_hash, frozenset(self.additional_info.items()) if self.additional_info else None))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelTag):
            return False
        
        return (self.start_time, self.end_time, self.text, self.source_media, self.model_track, self.frame_info, self.additional_info) == (other.start_time, other.end_time, other.text, other.source_media, other.model_track, other.frame_info, other.additional_info)
    
@dataclass(frozen=True)
class ContainerOutput:
    tags: list[ModelTag]
    progress: list[Progress]
    errors: list[Error]