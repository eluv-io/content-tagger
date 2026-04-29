
from dataclasses import dataclass

from src.tagging.fabric_tagging.model import JobRunStatus, TagArgs

"""structs for get status by content"""

@dataclass(frozen=True)
class ModelStatus:
    model: str
    track: str
    last_run: str
    percent_completion: float


@dataclass(frozen=True)
class ContentStatusResponse:
    models: list[ModelStatus]

"""structs for get status by content + model"""

@dataclass(frozen=True)
class JobUploadStatusSummary:
    num_job_parts: int
    num_tagged_parts: int


@dataclass(frozen=True)
class JobDetail:
    time_ran: str
    source_qid: str
    params: TagArgs
    job_status: JobRunStatus
    upload_status: JobUploadStatusSummary | None


@dataclass(frozen=True)
class ModelStatusSummary:
    model: str
    track: str
    last_run: str
    tagging_progress: float
    num_content_parts: int


@dataclass(frozen=True)
class ModelStatusResponse:
    summary: ModelStatusSummary
    jobs: list[JobDetail]
