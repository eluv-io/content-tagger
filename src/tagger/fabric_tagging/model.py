    

from dataclasses import dataclass, field
from typing import Literal
import time

from src.common.content import Content
from src.fetch.model import AssetScope, VideoScope

@dataclass
class RunConfig:
    # model config, used to overwrite the model level config
    model: dict = field(default_factory=dict)
    # stream name to run the model on, None to use the default stream. "assets" is a special case which will tag image assets
    stream: str | None = None

@dataclass
class TagArgs:
    features: dict[str, RunConfig]
    scope: AssetScope | VideoScope
    replace: bool

    def __str__(self):
        return f"TagArgs(features={self.features}, scope={self.scope}, replace={self.replace})"

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
class JobArgs:
    q: Content
    feature: str
    replace: bool
    runconfig: RunConfig
    scope: AssetScope | VideoScope

    def __str__(self):
        return f"JobArgs(q={self.q}, feature={self.feature}, replace={self.replace}, runconfig={self.runconfig}, scope={self.scope})"

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