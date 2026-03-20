from dataclasses import dataclass
from typing import Literal

from src.service.model import TagDetails
from src.tagging.fabric_tagging.model import TagArgs

job_status = Literal["queued", "running", "succeeded", "failed", "cancelled"]

@dataclass
class JobStoreConfig:
    base_url: str

@dataclass
class QueueItem:
    id: str
    qid: str
    created_at: float
    params: TagArgs
    status: job_status
    status_details: TagDetails | None
    error: str | None
    stop_requested: bool
    auth: str
    user: str
    tenant: str
    additional_info: dict

@dataclass
class CreateQueueItem:
    qid: str
    params: TagArgs
    status: job_status
    status_details: TagDetails | None
    additional_info: dict

@dataclass
class ListJobArgs:
    status: job_status | None = None
    qid: str | None = None
    user: str | None = None
    tenant: str | None = None

@dataclass
class UpdateJobRequest:
    id: str
    status: job_status
    status_details: TagDetails | None = None
    error: str | None = None