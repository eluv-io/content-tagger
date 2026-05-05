from dataclasses import dataclass, field
from typing import Literal


SystemResources = dict[str, int]

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
    content_aligned: bool = False
    # tagstore track dependencies
    dependencies: list[str] = field(default_factory=list)