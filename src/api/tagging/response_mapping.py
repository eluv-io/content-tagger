
from datetime import datetime

from src.service.model import *
from src.api.tagging.request_format import StatusRequest
from src.api.tagging.response_format import StatusResponse, StatusMeta, JobStatus, StopStatus, StopTaggingResponse, TagDetails
from src.common.logging import logger

_STATUS_SORT_ORDER = ["running", "queued", "failed", "succeeded", "cancelled"]

def _status_sort_key(job: TagJobStatusReport) -> tuple:
    try:
        return (_STATUS_SORT_ORDER.index(job.status), -job.created_at)
    except ValueError:
        return (len(_STATUS_SORT_ORDER), -job.created_at)

def map_all_jobs_status_to_response(
    all_jobs_status: list[TagJobStatusReport],
    req: StatusRequest,
) -> StatusResponse:
    # filter
    filtered = list(all_jobs_status)
    if req.status is not None:
        filtered = [j for j in filtered if j.status == req.status]
    if req.tenant is not None:
        filtered = [j for j in filtered if j.tenant == req.tenant]
    if req.user is not None:
        filtered = [j for j in filtered if j.user == req.user]

    logger.debug(f"Filtered jobs status from {len(all_jobs_status)} to {len(filtered)} based on query parameters", feature="status_filtering", total=len(all_jobs_status), filtered=len(filtered), status=req.status, tenant=req.tenant, user=req.user)

    # sort
    filtered.sort(key=_status_sort_key)

    total = len(filtered)

    # paginate
    paginated = filtered[req.start:] if req.limit is None else filtered[req.start:req.start + req.limit]

    return StatusResponse(
        jobs=[map_job_status_to_response(job_status) for job_status in paginated],
        meta=StatusMeta(
            total=total,
            start=req.start,
            limit=req.limit,
            count=len(paginated),
        ),
    )

def map_job_status_to_response(js: TagJobStatusReport) -> JobStatus:
    # convert float (seconds) to ISO string
    created_at = datetime.fromtimestamp(js.created_at).isoformat()
    return JobStatus(
        job_id=str(js.job_id),
        model=js.model,
        status=js.status,
        created_at=created_at,
        params=js.params,
        tenant=js.tenant,
        user=js.user,
        tag_details=TagDetails(
            tag_status=js.tagger_details.tag_status,
            stream=js.tagger_details.stream,
            time_running=js.tagger_details.time_running,
            tagging_progress=js.tagger_details.tagging_progress,
            failed=js.tagger_details.failed,
        ) if js.tagger_details else None,
    )

def map_stop_results_to_response(stop_results: list[TagStopResult]) -> StopTaggingResponse:
    num_results = len(stop_results)
    return StopTaggingResponse(
        jobs=[StopStatus(job_id=str(result.job_id), message=result.message) for result in stop_results],
        message=f"Stopping {num_results} job(s). Check with /status for completion."
    )