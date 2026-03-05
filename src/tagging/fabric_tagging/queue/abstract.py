from typing import Protocol

from src.tagging.fabric_tagging.model import *
from src.tagging.fabric_tagging.queue.model import *

class JobStore(Protocol):
    def create_job(self, qid: str, params: TagArgs) -> None:
        ...
    
    def list_jobs(self, args: ListJobArgs) -> list[QueueItem]:
        ...

    def update_job(self, args: UpdateJobRequest) -> None:
        ...