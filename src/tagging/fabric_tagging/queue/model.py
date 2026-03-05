from dataclasses import dataclass
from typing import Literal

from src.tagging.fabric_tagging.model import TagArgs, TagJobStatusReport

job_status = Literal["queued", "running", "succeeded", "failed", "cancelled"]

@dataclass
class QueueItem:
    id: str
    qid: str
    params: TagArgs
    auth: str
    user: str
    tenant: str

@dataclass
class CreateQueueItem:
    qid: str
    params: TagArgs
    status: job_status
    status_details: TagJobStatusReport

@dataclass
class ListJobArgs:
    qid: str | None = None
    user: str | None = None
    tenant: str | None = None

@dataclass
class UpdateJobRequest:
    id: str
    status: job_status
    status_details: TagJobStatusReport