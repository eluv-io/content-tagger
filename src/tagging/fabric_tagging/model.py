    

from dataclasses import dataclass
from typing import Literal

from src.common.content import Content
from src.fetch.model import Scope
from src.tagging.uploading.config import UploaderConfig

@dataclass
class FabricTaggerConfig:
    media_dir: str
    uploader: UploaderConfig

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
    message: str

@dataclass(frozen=True)
class TagJobStatusReport:
    status: JobStateDescription
    time_running: float
    tagging_progress: str
    missing_tags: list[str]
    failed: list[str]
    model: str
    stream: str
    message: str | None = None