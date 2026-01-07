from dataclasses import dataclass, field

@dataclass
class ModelParams:
    stream: str | None = None
    model: dict = field(default_factory=dict)

@dataclass
class BaseTagAPIArgs:
    features: dict[str, ModelParams]
    destination_qid: str = ""
    replace: bool = False

@dataclass
class TagAPIArgs(BaseTagAPIArgs):
    start_time: int = 0
    end_time: int = 10**16

@dataclass
class ImageTagAPIArgs(BaseTagAPIArgs):
    assets: list[str] | None = None

@dataclass
class LiveTagAPIArgs(BaseTagAPIArgs):
    segment_length: int = 4
    max_duration: int | None = None