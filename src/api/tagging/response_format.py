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
class JobStatus:
    job_id: str
    status: str
    created_at: str
    time_running: float
    tagging_progress: str
    failed: list[str]
    model: str
    stream: str

@dataclass(frozen=True)
class StatusResponse:
    jobs: list[JobStatus]

@dataclass(frozen=True)
class StopStatus:
    job_id: str
    message: str

@dataclass(frozen=True)
class StopTaggingResponse:
    jobs: list[StopStatus]
    message: str