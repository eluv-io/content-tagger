    

from dataclasses import dataclass
from typing import Literal

from src.common.content import Content
from src.fetch.model import Scope
from src.tag_containers.model import ContainerInfo

@dataclass
class FabricTaggerConfig:
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
    "Fetching content",
    "Tagging content",
    "Completed",
    "Failed",
    "Stopped"
]

@dataclass
class JobArgs(TagArgs):
    q: Content
    retry_upload: bool

@dataclass
class JobID:
    """
    Unique identifier for a job to prevent duplication.
    """
    qhit: str
    feature: str
    # TODO: weirdness
    stream: str

    def __hash__(self):
        return hash((self.qhit, self.feature, self.stream))
    
    def __str__(self):
        return f"(qid={self.qhit}, model={self.feature}, stream={self.stream})"
    
@dataclass(frozen=True)
class TagStartResult:
    started: bool
    job_id: JobID
    message: str

@dataclass(frozen=True)
class TagJobStatusReport:
    job_id: JobID
    status: JobStateDescription
    time_running: float
    tagging_progress: str
    missing_tags: list[str]
    failed: list[str]
    model: str
    stream: str
    message: str | None = None

@dataclass(frozen=True)
class TagStopResult:
    job_id: JobID
    message: str

@dataclass(frozen=True)
class UploadStatus:
    all_sources: list[str]
    tagged_sources: list[str]

@dataclass(frozen=True)
class JobRunStatus:
    status: str
    
@dataclass(frozen=True)
class TagContentStatusReport:
    source_qid: str
    params: TagArgs
    container: ContainerInfo
    job_status: JobRunStatus
    upload_status: UploadStatus | None