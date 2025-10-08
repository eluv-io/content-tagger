from dataclasses import dataclass
from typing import Any, Literal

from src.common.content import Content
from src.tagger.fabric_tagging.model import *

@dataclass
class TagRequest:
    q: Content
    args: TagArgs

    def __str__(self):
        return f"TagRequest(q={self.q}, args={self.args})"

@dataclass
class StatusRequest:
    qhit: str

    def __str__(self):
        return f"StatusRequest(qhit={self.qhit})"

@dataclass
class StopRequest:
    qhit: str
    feature: str | None
    stream: str | None
    status: Literal["Stopped", "Failed", "Completed"]

    def __str__(self):
        return f"StopRequest(qhit={self.qhit}, feature={self.feature}, stream={self.stream}, status={self.status})"

@dataclass
class JobTransition:
    """
    Request to transition the job to the next stage
    """

    job_id: JobID
    # depends on the job stage, some stages require information to be passed forward
    data: Any

    def __str__(self):
        return f"JobTransition(job_id={self.job_id}, data={self.data})"

@dataclass
class UploadTick:
    def __str__(self):
        return "UploadTick()"

@dataclass
class CleanupRequest:
    def __str__(self):
        return "CleanupRequest()"

Request = TagRequest | StatusRequest | StopRequest | JobTransition | CleanupRequest | UploadTick