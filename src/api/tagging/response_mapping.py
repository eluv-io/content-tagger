
from datetime import datetime

from src.service.model import *
from src.api.tagging.request_format import StatusRequest
from src.api.tagging.response_format import StatusResponse, StatusMeta, JobStatus, StopStatus, StopTaggingResponse, TagDetails
from src.common.logging import logger

_STATUS_SORT_ORDER = ["running", "queued", "failed", "succeeded", "cancelled"]

def _status_sort_key(job: TagJobStatusResult) -> tuple:
    try:
        return (_STATUS_SORT_ORDER.index(job.status), -job.created_at)
    except ValueError:
        return (len(_STATUS_SORT_ORDER), -job.created_at)

def map_all_jobs_status_to_response(
    all_jobs_status: list[TagJobStatusResult],
    req: StatusRequest,
) -> StatusResponse:
    # filter
    filtered = list(all_jobs_status)
    if req.status is not None:
        filtered = [j for j in filtered if j.status == req.status]
    if req.model is not None:
        filtered = [j for j in filtered if j.model == req.model]

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

def map_job_status_to_response(js: TagJobStatusResult) -> JobStatus:
    # convert float (seconds) to ISO string
    created_at = datetime.fromtimestamp(js.created_at).isoformat()

    # legacy compatibility
    tagging_progress = js.tagger_details.tagging_progress if js.tagger_details else "0/0"
    return JobStatus(
        qid=js.qid,
        job_id=str(js.job_id),
        model=js.model,
        status=js.status,
        created_at=created_at,
        stream=js.stream,
        title=js.title,
        params=js.params,
        tenant=js.tenant,
        user=js.user,
        error=js.error,
        tagging_progress=tagging_progress,
        tag_details=js.tagger_details,
    )

def map_stop_results_to_response(stop_results: list[TagStopResult]) -> StopTaggingResponse:
    num_results = len(stop_results)
    return StopTaggingResponse(
        jobs=[StopStatus(job_id=str(result.job_id), message=result.message) for result in stop_results],
        message=f"Stopping {num_results} job(s). Check with /status for completion."
    )