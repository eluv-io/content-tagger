from dataclasses import dataclass

@dataclass(frozen=True)
class StartStatus:
    job_id: str
    model: str
    started: bool
    message: str
    error: str | None

@dataclass(frozen=True)
class StartTaggingResponse:
    jobs: list[StartStatus]

@dataclass(frozen=True)
class TagDetails:
    tag_status: str
    stream: str
    time_running: float
    tagging_progress: str
    failed: list[str]

@dataclass(frozen=True)
class JobStatus:
    job_id: str
    status: str
    created_at: str
    model: str
    params: dict
    tenant: str
    user: str
    tag_details: TagDetails | None

@dataclass(frozen=True)
class StatusMeta:
    total: int
    start: int
    limit: int | None
    count: int

@dataclass(frozen=True)
class StatusResponse:
    jobs: list[JobStatus]
    meta: StatusMeta

@dataclass(frozen=True)
class StopStatus:
    job_id: str
    message: str

@dataclass(frozen=True)
class StopTaggingResponse:
    jobs: list[StopStatus]
    message: str