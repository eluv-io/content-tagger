from dataclasses import dataclass

from src.tagging.fabric_tagging.model import TagArgs, TagJobStatusReport

@dataclass
class QueueItem:
    id: str
    qid: str
    params: TagArgs
    user: str
    tenant: str

@dataclass
class CreateQueueItem:
    qid: str
    params: TagArgs
    status: str
    status_details: TagJobStatusReport

@dataclass
class ListJobArgs:
    qid: str | None = None
    user: str | None = None
    tenant: str | None = None

@dataclass
class UpdateJobRequest:
    id: str
    status: str
    status_details: TagJobStatusReport

