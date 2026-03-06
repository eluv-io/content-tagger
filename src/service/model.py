from dataclasses import dataclass

@dataclass(frozen=True)
class TagStartResult:
    job_id: str
    started: bool
    message: str
    created_at: float

@dataclass(frozen=True)
class TagJobStatusReport:
    job_id: str
    status: str
    created_at: float
    time_running: float
    tagging_progress: str
    failed: list[str]
    model: str
    stream: str
    message: str | None = None

@dataclass(frozen=True)
class TagStopResult:
    job_id: str
    message: str