    

from dataclasses import dataclass, field
from typing import Literal
import time

from src.common.content import Content
from src.fetch.model import Scope

@dataclass
class FabricTaggerConfig:
    media_dir: str

@dataclass
class TagArgs:
    feature: str
    run_config: dict
    scope: Scope
    replace: bool

    def __str__(self):
        return f"TagArgs(run_config={self.run_config}, scope={self.scope}, replace={self.replace})"

JobStateDescription = Literal[
    "Fetching content",
    "Tagging content",
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
            status="Fetching content",
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
    runconfig: dict
    scope: Scope
    retry_fetch: bool

    def __str__(self):
        return f"JobArgs(q={self.q}, feature={self.feature}, replace={self.replace}, runconfig={self.runconfig}, scope={self.scope}, retry_fetch={self.retry_fetch})"

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
        return f"JobID(qhit={self.qhit}, feature={self.feature}, stream={self.stream})"