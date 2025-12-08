from dataclasses import dataclass
from typing_extensions import Literal

from src.tags.tagstore.model import Tag
from src.common.model import SystemResources

MediaInput = list[str] | str

@dataclass
class ModelConfig:
    """
    Describes static attributes of a model
    """
    image: str
    type: Literal["audio", "video", "frame"]
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
    # destination path for the tags
    tags_dir: str
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
    live: bool
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
class ModelTag:
    """
    Represents a tag produced at the model level
    """
    start_time: int
    end_time: int
    text: str
    frame_tags: dict
    source_media: str
    track: str

    def __hash__(self) -> int:
        return hash((self.start_time, self.end_time, self.text, self.source_media, self.track))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelTag):
            return False
        return (self.start_time, self.end_time, self.text, self.source_media, self.track) == (other.start_time, other.end_time, other.text, other.source_media, other.track)

@dataclass
class FrameTag:
    frame_idx: str
    confidence: float
    box: dict
    text: str