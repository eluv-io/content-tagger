from dataclasses import asdict, fields

from src.tagging.fabric_tagging.model import *
from src.api.tagging.response_format import StatusResponse, JobStatus

def map_all_jobs_status_to_response(all_jobs_status: list[TagJobStatusReport]) -> StatusResponse:
    return StatusResponse(
        jobs=[map_job_status_to_response(job_status) for job_status in all_jobs_status]
    )

def map_job_status_to_response(js: TagJobStatusReport) -> JobStatus:
    return JobStatus(
        job_id=str(js.job_id),
        model=js.model,
        stream=js.stream,
        status=js.status,
        time_running=js.time_running,
        tagging_progress=js.tagging_progress,
        missing_tags=js.missing_tags,
        failed=js.failed,
    )