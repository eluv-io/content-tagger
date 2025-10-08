    

from dataclasses import dataclass, field
from typing import Literal
import time
import threading

from src.common.content import Content
from src.fetch.model import AssetScope, VideoScope
from src.tag_containers.containers import TagContainer
from src.fetch.fetch_content import DownloadResult

JobStateDescription = Literal[
    "Starting",
    "Fetching content",
    "Tagging content",
    "Uploading tags",
    "Completed",
    "Failed",
    "Stopped"
]

@dataclass
class JobStatus:
    status: JobStateDescription
    time_started: float
    time_ended: float | None
    tagging_progress: str
    failed: list[str]

    @staticmethod
    def starting() -> 'JobStatus':
        return JobStatus(
            status="Starting",
            time_started=time.time(),
            time_ended=None,
            tagging_progress="0%",
            failed=[]
        )

@dataclass
class RunConfig:
    # model config, used to overwrite the model level config
    model: dict = field(default_factory=dict)
    # stream name to run the model on, None to use the default stream. "assets" is a special case which will tag image assets
    stream: str | None = None

@dataclass
class JobArgs:
    q: Content
    feature: str
    replace: bool
    runconfig: RunConfig
    scope: AssetScope | VideoScope

    def __str__(self):
        return f"JobArgs(q={self.q}, feature={self.feature}, replace={self.replace}, runconfig={self.runconfig}, scope={self.scope})"

@dataclass
class JobState:
    # everything that might change during the job
    status: JobStatus
    taghandle: str
    uploaded_sources: list[str]
    message: str
    media: DownloadResult | None
    container: TagContainer | None

    @staticmethod
    def starting() -> 'JobState':
        return JobState(
            status=JobStatus.starting(),
            taghandle="",
            uploaded_sources=[],
            message="",
            media=None,
            container=None
        )

@dataclass
class TagJob:
    args: JobArgs
    state: JobState
    upload_job: str
    stop_event: threading.Event
    tagging_done: threading.Event | None

    def get_id(self) -> 'JobID':
        assert self.args.runconfig.stream is not None
        return JobID(qhit=self.args.q.qhit, feature=self.args.feature, stream=self.args.runconfig.stream)

@dataclass
class JobID:
    """
    Unique identifier for a job to prevent duplication.
    """
    qhit: str
    feature: str
    stream: str

    def __hash__(self):
        return hash((self.qhit, self.feature, self.stream))
    
    def __str__(self):
        return f"JobID(qhit={self.qhit}, feature={self.feature}, stream={self.stream})"

@dataclass
class JobStore:
    active_jobs: dict[JobID, TagJob] = field(default_factory=dict)
    inactive_jobs: dict[JobID, TagJob] = field(default_factory=dict)

@dataclass
class TagArgs:
    features: dict[str, RunConfig]
    scope: AssetScope | VideoScope
    replace: bool

    def __str__(self):
        return f"TagArgs(features={self.features}, scope={self.scope}, replace={self.replace})"