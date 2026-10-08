from dataclasses import dataclass
from typing import Literal

from src.service.model import TagDetails
from src.tagging.fabric_tagging.model import TagArgs

job_status = Literal["pending", "queued", "running", "cancelling", "succeeded", "failed", "cancelled"]

@dataclass
class JobStoreConfig:
    # queue manager url; if empty, jobs are stored as files under base_dir
    base_url: str = ""
    base_dir: str = ""
    # sent as X-Worker-Secret
    worker_secret: str | None = None
    timeout: int = 30
    # queue manager job type the tagger's jobs are stored as
    job_type: str = "tag"

@dataclass
class QueueItem:
    id: str
    qid: str
    model: str
    created_at: float
    # None while the job is pending
    params: TagArgs | None
    status: job_status
    status_details: TagDetails | None
    error: str | None
    auth: str
    user: str
    tenant: str
    additional_info: dict

@dataclass
class CreateQueueItem:
    qid: str
    model: str

@dataclass
class ReleaseJobRequest:
    id: str
    params: TagArgs
    deps: list[str]
    additional_info: dict

@dataclass
class ListJobArgs:
    status: job_status | None = None
    qid: str | None = None
    user: str | None = None
    tenant: str | None = None
    # include queued jobs that can't be claimed yet: unmet dependencies or their resource in use
    include_unready: bool = False
    limit: int | None = None

@dataclass
class CompleteJobRequest:
    id: str
    # succeeded/failed for a running or pending job, cancelled for a cancelling one
    status: job_status
    status_details: TagDetails | None = None
    error: str | None = None
