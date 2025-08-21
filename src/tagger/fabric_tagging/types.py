    
from dataclasses import dataclass, field
from typing import Literal
import time
import threading

from src.common.content import Content
from src.tagger.model_containers.containers import TagContainer

JobState = Literal[
    "Starting",
    "Fetching content",
    "Waiting for resources",
    "Tagging content",
    "Completed",
    "Failed",
    "Stopped"
]

@dataclass
class JobStatus:
    status: JobState
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
    model: dict
    # stream name to run the model on, None to use the default stream. "image" is a special case which will tag image assets
    stream: str

@dataclass
class JobArgs:
    q: Content
    feature: str
    replace: bool
    runconfig: RunConfig
    start_time: int | None
    end_time: int | None

@dataclass
class Job:
    args: JobArgs
    status: JobStatus
    taghandle: str
    lock: threading.Lock
    stopevent: threading.Event
    uploaded_sources: list[str]
    container: TagContainer | None

    def get_id(self) -> 'JobID':
        return JobID(qhit=self.args.q.qhit, feature=self.args.feature, stream=self.args.runconfig.stream)

@dataclass
class JobID:
    qhit: str
    feature: str
    stream: str

    def __hash__(self):
        return hash((self.qhit, self.feature, self.stream))

@dataclass
class JobStore:
    active_jobs: dict[JobID, Job] = field(default_factory=dict)
    inactive_jobs: dict[JobID, Job] = field(default_factory=dict)
