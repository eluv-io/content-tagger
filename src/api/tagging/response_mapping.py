from dataclasses import asdict, fields

from src.tagging.fabric_tagging.model import *
from src.api.tagging.response_format import StatusResponse, JobStatus

def map_all_jobs_status_to_response(all_jobs_status: list[TagJobStatusReport]) -> StatusResponse:
    return StatusResponse(
        jobs=[map_job_status_to_response(job_status) for job_status in all_jobs_status]
    )

def map_job_status_to_response(job_status: TagJobStatusReport) -> JobStatus:
    valid_fields = {f.name for f in fields(JobStatus)}
    job_dict = asdict(job_status)
    filtered_dict = {k: v for k, v in job_dict.items() if k in valid_fields}
    return JobStatus(**filtered_dict)