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
    qid: str

    def __str__(self):
        return f"StatusRequest(qid={self.qid})"

@dataclass
class StopRequest:
    qid: str
    feature: str | None
    status: Literal["Stopped", "Failed", "Completed"]
    error: Exception | None = None

    def __str__(self):
        return f"StopRequest(qid={self.qid}, feature={self.feature}, status={self.status}, error={self.error})"

@dataclass
class EnterFetchingPhase:
    """Request to enter fetching phase"""
    job_id: JobID

    def __str__(self):
        return f"EnterFetchingPhase(job_id={self.job_id})"

@dataclass
class EnterTaggingPhase:
    """Request to enter tagging phase."""
    job_id: JobID
    dl_result: DownloadResult

    def __str__(self):
        return f"EnterTaggingPhase(job_id={self.job_id})"

@dataclass
class EnterCompletePhase:
    """Request to enter complete phase to signal job completion."""
    job_id: JobID

    def __str__(self):
        return f"EnterCompletePhase(job_id={self.job_id})"

@dataclass
class UploadTick:
    """Request to trigger an upload tick for tags to be uploaded to the tagstore."""
    def __str__(self):
        return "UploadTick()"

@dataclass
class CleanupRequest:
    def __str__(self):
        return "CleanupRequest()"

Request = TagRequest | StatusRequest | StopRequest | EnterFetchingPhase | EnterTaggingPhase | EnterCompletePhase | CleanupRequest | UploadTick