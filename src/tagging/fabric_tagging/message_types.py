from dataclasses import dataclass
from typing import Literal

from src.common.content import Content
from src.tagging.fabric_tagging.model import *
from src.fetch.model import DownloadResult

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
class EnterFetchingPhase:
    """Request to enter fetching phase"""
    job_id: JobID

    def __str__(self):
        return f"EnterFetchingPhase(job_id={self.job_id})"

@dataclass
class EnterTaggingPhase:
    """Request to enter tagging phase"""
    job_id: JobID
    data: DownloadResult

    def __str__(self):
        return f"EnterTaggingPhase(job_id={self.job_id})"

@dataclass
class EnterCompletePhase:
    """Request to enter complete phase"""
    job_id: JobID

    def __str__(self):
        return f"EnterCompletePhase(job_id={self.job_id})"

@dataclass
class UploadTick:
    def __str__(self):
        return "UploadTick()"

@dataclass
class CleanupRequest:
    def __str__(self):
        return "CleanupRequest()"

Request = TagRequest | StatusRequest | StopRequest | EnterFetchingPhase | EnterTaggingPhase | EnterCompletePhase | CleanupRequest | UploadTick