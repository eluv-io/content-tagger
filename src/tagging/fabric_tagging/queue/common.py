def job_uid(qid: str, model: str) -> str:
    """Unique among active jobs: creating a job is refused while another with its uid hasn't ended, so a model is
    active at most once at a time on a content object. Stored as the queue manager's resource_hash."""
    return f"{qid}/{model}"
