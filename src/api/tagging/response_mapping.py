from dataclasses import asdict, fields

from src.tagging.fabric_tagging.model import *
from src.api.tagging.response_format import StatusResponse, JobStatus, StopStatus, StopTaggingResponse

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

def map_stop_results_to_response(stop_results: list[TagStopResult]) -> StopTaggingResponse:
    num_results = len(stop_results)
    return StopTaggingResponse(
        jobs=[StopStatus(job_id=str(result.job_id), message=result.message) for result in stop_results],
        message=f"Stopping {num_results} job(s). Check with /status for completion."
    )