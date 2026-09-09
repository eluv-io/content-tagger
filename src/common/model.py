from dataclasses import dataclass, field
from typing import Literal


SystemResources = dict[str, int]

@dataclass
class TagsConfig:
    # when true, tracks created for this model's outputs are marked hidden
    hidden: bool = False

@dataclass
class ModelConfig:
    """
    Describes static attributes of a model
    """
    image: str
    # empty description string indicates that it will be hidden from listing API
    description: str
    type: Literal["audio", "video", "frame", "processor"]
    resources: SystemResources
    # indicates that the model will output tags aligned to the full content rather than individual parts
    scope: dict | None = None
    content_aligned: bool = False
    track_outputs: list[str] = field(default_factory=list)
    track_dependencies: list[str] = field(default_factory=list)
    tags: TagsConfig = field(default_factory=TagsConfig)