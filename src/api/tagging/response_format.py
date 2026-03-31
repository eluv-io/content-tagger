from dataclasses import dataclass

from src.service.model import TagDetails

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
class JobStatus: 
    qid: str
    job_id: str
    status: str
    model: str
    stream: str
    created_at: str
    params: dict
    tenant: str
    user: str
    title: str
    tagging_progress: str
    # between 0 and 1
    progress: float
    error: str | None
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