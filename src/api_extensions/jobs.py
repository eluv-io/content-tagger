
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import UpdateJobRequest


def delete_job(id: str, auth: str, js: JobStore) -> None:
    js.update_job(UpdateJobRequest(id=id, status="deleted"), auth)
