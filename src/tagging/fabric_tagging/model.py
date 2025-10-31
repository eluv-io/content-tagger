    

from dataclasses import dataclass
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
    # arbitrary params to pass to the model container
    run_config: dict
    # scope of the tagging job w.r.t the content
    scope: Scope
    # if false, then the tagger will ignore tagging of a media source if the (source, model) pair
    # already exists in the tagstore.
    replace: bool
    destination_qid: str

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
class JobArgs(TagArgs):
    q: Content
    # whether to retry fetching content if it fails or end the tagging job. 
    # important for the livestream case where momentary failures can be expected.
    retry_fetch: bool

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