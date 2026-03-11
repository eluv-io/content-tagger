from dataclasses import dataclass

@dataclass(frozen=True)
class StatusArgs:
    qid: str | None
    user: str | None
    tenant: str | None
    title: str | None

@dataclass(frozen=True)
class TagStartResult:
    job_id: str
    started: bool
    message: str
    created_at: float

@dataclass(frozen=True)
class TagDetails:
    tag_status: str
    stream: str
    time_running: float
    tagging_progress: str
    failed: list[str]

@dataclass(frozen=True)
class TagJobStatusReport:
    qid: str
    job_id: str
    status: str
    model: str
    created_at: float
    params: dict
    tenant: str = ""
    user: str = ""
    title: str = ""
    message: str | None = None
    tagger_details: TagDetails | None = None

@dataclass(frozen=True)
class TagStopResult:
    job_id: str
    message: str