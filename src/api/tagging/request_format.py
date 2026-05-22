
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypeAlias

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
    track_suffix: str = ""
    caller_info: dict[str, str] = field(default_factory=dict)
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