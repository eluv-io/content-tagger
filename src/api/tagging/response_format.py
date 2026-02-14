from dataclasses import dataclass

@dataclass(frozen=True)
class StartStatus:
    job_id: str
    model: str
    stream: str
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
    time_running: float
    tagging_progress: str
    missing_tags: list[str]
    failed: list[str]
    model: str
    stream: str

@dataclass(frozen=True)
class StatusResponse:
    jobs: list[JobStatus]