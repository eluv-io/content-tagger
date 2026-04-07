from typing import Protocol

from src.tagging.fabric_tagging.model import *
from src.tagging.fabric_tagging.queue.model import *

class JobStore(Protocol):
    def create_job(self, args: CreateQueueItem, auth: str) -> QueueItem:
        ...

    def claim_job(self, id: str, auth: str) -> bool:
        ...

    def get_job(self, id: str) -> QueueItem:
        ...

    def list_jobs(self, args: ListJobArgs, auth: str) -> list[QueueItem]:
        ...

    def update_job(self, args: UpdateJobRequest, auth: str) -> None:
        ...

    def stop_job(self, id: str, auth: str) -> None:
        ...