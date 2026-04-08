from dataclasses import dataclass, field
from typing_extensions import Literal

from src.common.content import Content
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
    description: str
    type: Literal["audio", "video", "frame", "processor"]
    resources: SystemResources
    content_aligned: bool = False

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
    media_dir: str
    # path to the JSONL output file
    output_path: str
    # runtime params passed to the model (unique schema per model)
    run_config: dict
    # mount the qid+auth in case the container needs API calls
    q: Content
    # static attributes of the container to run
    model_config: ModelConfig

@dataclass
class ContainerRequest:
    """
    Args struct for getting a TagContainer from the ContainerFactory
    """

    # e.g: "caption", "asr" (same model handle used in the API)
    model_id: str
    # directory on host filesystem where media will arrive
    media_dir: str
    # runtime params passed to the model
    run_config: dict
    # mount the qid+auth in case the container needs API calls
    q: Content
    # handle passed to the container for tracking
    job_id: str | None

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