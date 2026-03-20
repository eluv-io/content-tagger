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
    time_running: float
    # between 0 and 1
    progress: float
    # legacy
    tagging_progress: str

    # extra detail
    total_parts: int
    downloaded_parts: int
    tagged_parts: int

@dataclass(frozen=True)
class TagJobStatusResult: 
    qid: str
    job_id: str
    status: str
    model: str
    stream: str
    created_at: float
    params: dict
    tenant: str
    user: str
    title: str
    error: str | None
    tagger_details: TagDetails | None

@dataclass(frozen=True)
class TagStopResult:
    job_id: str
    message: str