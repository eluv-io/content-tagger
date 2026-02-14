from dataclasses import dataclass

@dataclass(frozen=True)
class JobStatus:
    status: str
    time_running: float
    tagging_progress: str
    missing_tags: list[str]
    failed: list[str]
    model: str
    stream: str

@dataclass
class StatusResponse:
    jobs: list[JobStatus]