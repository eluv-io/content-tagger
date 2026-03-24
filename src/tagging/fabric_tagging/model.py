    

import time
from dataclasses import dataclass
from typing import Literal

from src.common.content import Content
from src.fetch.model import Scope
from src.tag_containers.model import ContainerInfo

@dataclass
class TaggerWorkerConfig:
    media_dir: str

@dataclass
class TagArgs:
    feature: str
    # arbitrary params to pass to the model container
    run_config: dict
    # scope of the tagging job w.r.t the content
    scope: Scope
    replace: bool
    destination_qid: str
    max_fetch_retries: int

JobStateDescription = Literal[
    "Queued",
    "Fetching content",
    "Tagging content",
    "Completed",
    "Failed",
    "Stopped"
]

@dataclass(frozen=True)
class JobStatus:
    status: JobStateDescription
    time_started: float
    time_ended: float | None
    total_sources: list[str]
    downloaded_sources: list[str]
    tagged_sources: list[str]
    uploaded_sources: list[str]
    warnings: list[str]
    error: str | None

    @staticmethod
    def starting() -> 'JobStatus':
        return JobStatus(
            status="Fetching content",
            time_started=time.time(),
            time_ended=None,
            total_sources=[],
            tagged_sources=[],
            downloaded_sources=[],
            uploaded_sources=[],
            warnings=[],
            error=None,
        )

@dataclass
class JobArgs(TagArgs):
    q: Content
    retry_upload: bool

@dataclass
class JobID:
    """
    Unique identifier for a job to prevent duplication.
    """
    qid: str
    feature: str
    # TODO: weirdness
    stream: str

    def __hash__(self):
        return hash((self.qid, self.feature, self.stream))
    
    def __str__(self):
        return f"(qid={self.qid}, model={self.feature}, stream={self.stream})"
    
@dataclass(frozen=True)
class TagStartResult:
    started: bool
    job_id: JobID
    message: str

@dataclass(frozen=True)
class TagStopResult:
    job_id: JobID
    message: str

@dataclass(frozen=True)
class UploadStatus:
    all_sources: list[str]
    downloaded_sources: list[str]
    tagged_sources: list[str]
    uploaded_sources: list[str]

@dataclass(frozen=True)
class JobRunStatus:
    status: str
    time_ran: str
    
@dataclass(frozen=True)
class TagContentStatusReport:
    source_qid: str
    params: TagArgs
    container: ContainerInfo
    job_status: JobRunStatus
    upload_status: UploadStatus | None

@dataclass(frozen=True)
class TagStatusResult:
    status: JobStatus
    model: str
    stream: str