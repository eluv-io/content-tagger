from dataclasses import dataclass

from src.tagging.fabric_tagging.model import TagArgs, JobRunStatus


@dataclass(frozen=True)
class JobUploadStatusSummary:
    num_job_parts: int
    num_tagged_parts: int


@dataclass(frozen=True)
class JobDetail:
    time_ran: str
    source_qid: str
    params: TagArgs | None
    job_status: JobRunStatus | None
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
