from typing import Protocol

from src.tagging.fabric_tagging.model import *
from src.tagging.fabric_tagging.queue.model import *

class JobStore(Protocol):
    def create_job(self, args: CreateQueueItem, auth: str) -> QueueItem:
        """Create a job in the pending state. It isn't claimable until it is released."""
        ...

    def release_job(self, args: ReleaseJobRequest, auth: str) -> bool:
        """Set the params of a pending job and move it to queued. Returns False if the job is no longer pending."""
        ...

    def claim_job(self, id: str, auth: str) -> bool:
        ...

    def get_job(self, id: str) -> QueueItem:
        ...

    def list_jobs(self, args: ListJobArgs, auth: str) -> list[QueueItem]:
        ...

    def update_progress(self, id: str, status_details: TagDetails, auth: str) -> QueueItem:
        ...

    def complete_job(self, args: CompleteJobRequest, auth: str) -> bool:
        ...

    def cancel_job(self, id: str, auth: str, reason: str | None = None) -> None:
        ...

    def delete_job(self, id: str, auth: str) -> None:
        ...
