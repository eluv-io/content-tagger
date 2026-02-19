
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypeAlias

@dataclass
class ScopeVideo:
    type: Literal["video"] = "video"
    start_time: int = 0
    end_time: int = 10**16
    stream: str = ""

@dataclass
class ScopeProcessor:
    type: Literal["processor"] = "processor"
    start_time: int = 0
    end_time: int = 10**16
    chunk_size: int = 600
    stream: str = ""

@dataclass
class ScopeAssets:
    type: Literal["assets"] = "assets"
    assets: list[str] | None = None

@dataclass
class ScopeLivestream:
    type: Literal["livestream"] = "livestream"
    stream: str = ""
    segment_length: int = 4
    max_duration: Optional[int] = None

ScopeDTO: TypeAlias = ScopeVideo | ScopeProcessor | ScopeAssets | ScopeLivestream

@dataclass
class TaggerArgs:
    destination_qid: str = ""
    replace: bool = False
    max_fetch_retries: int = 3
    scope: dict[str, Any] = field(default_factory=dict)

@dataclass
class JobSpec:
    model: str
    model_params: dict[str, Any] = field(default_factory=dict)
    overrides: TaggerArgs | None = None

@dataclass
class StartJobsRequest:
    options: TaggerArgs = field(default_factory=TaggerArgs)
    jobs: list[JobSpec] = field(default_factory=list)