
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
class TaggerOptions:
    destination_qid: str | None = None
    replace: bool | None = None
    max_fetch_retries: int | None = None
    # unstructured dict to allow for flexible scope definitions - will be parsed into a ScopeDTO
    scope: dict[str, Any] = field(default_factory=dict)

@dataclass
class JobSpec:
    model: str
    model_params: dict[str, Any] = field(default_factory=dict)
    overrides: TaggerOptions = field(default_factory=TaggerOptions)

@dataclass
class StartJobsRequest:
    options: TaggerOptions = field(default_factory=TaggerOptions)
    jobs: list[JobSpec] = field(default_factory=list)

@dataclass
class StatusRequest:
    start: int = 0
    limit: int | None = None
    qid: str | None = None
    status: str | None = None
    tenant: str | None = None
    user: str | None = None
    model: str | None = None
    title: str | None = None