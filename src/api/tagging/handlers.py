from copy import deepcopy
import os
from common_ml.utils.metrics import timeit

from flask import request, current_app
from flask_smorest import Blueprint

from src.api.arg_resolver import ArgsResolver
from src.api_extensions.jobs import DeleteJobQuerySchema, DeleteJobRequest, delete_job
from src.api_extensions.models import ListingResponse, ListingResponseSchema, list_models
from src.common.model import ModelConfig
from src.service.abstract import TaggerService
from src.api.tagging.request_format import (
    StartJobsRequestSchema,
    StatusRequestSchema,
)
from src.api.tagging.response_format import (
    StartStatus,
    StartTaggingResponse,
    StartTaggingResponseSchema,
    StatusResponseSchema,
    StopTaggingResponseSchema,
)
from src.common.logging import logger

from src.common.errors import *
from src.api.auth import *
from src.common.content import Content
from src.api.tagging.request_mapping import *
from src.api.tagging.response_mapping import *
from src.service.impl.queue_based import QueueService
from src.status.get_info import UserInfoResolver
from src.tagging.fabric_tagging.queue.abstract import JobStore

tagging_blp = Blueprint(
    "Operate Tagging", __name__, description="Start, query and stop tagging jobs."
)


@tagging_blp.route("/<qid>/tag", methods=["POST"])
@tagging_blp.arguments(StartJobsRequestSchema)
@tagging_blp.response(200, StartTaggingResponseSchema)
def handle_tag(args: StartJobsRequest, qid: str) -> StartTaggingResponse:
    """Start Tagging
    
    Start a batch of tagging jobs for a content object. The request body contains a list of models to run as well as any global options or model specific runtime parameters.
    """
    q = authorize(qid, request)

    logger.debug(args)

    if args.options.destination_qid:
        authorize(args.options.destination_qid, request)

    if args.options.index_qid:
        authorize(args.options.index_qid, request)

    arg_resolver: ArgsResolver = current_app.config["state"]["arg_resolver"]

    with timeit("resolving tag args"):
        tag_args = arg_resolver.resolve(args, q)

    return _execute_tagging(q, tag_args)

def _execute_tagging(q: Content, tag_args: list[TagArgs]) -> StartTaggingResponse:
    """Execute tagging for multiple features and return start status response."""
    tagger: TaggerService = current_app.config["state"]["service"]

    jobs: list[StartStatus] = []

    start_results = tagger.tag(q, tag_args)

    for arg, result in zip(tag_args, start_results):
        jobs.append(
            StartStatus(
                job_id=result.job_id,
                model=arg.feature,
                started=result.started,
                message=result.message,
                dependencies=result.dependencies,
                error=None,
            )
        )

    return StartTaggingResponse(jobs=jobs)

@tagging_blp.route("/<qid>/job-status", methods=["GET"])
@tagging_blp.arguments(StatusRequestSchema, location="query")
@tagging_blp.response(200, StatusResponseSchema)
def handle_status_content(status_req: StatusRequest, qid: str) -> StatusResponse:
    """Get job statuses for a content object

    Get the status of all jobs for a content object. Requires the content's qid in the path and optional filters in the query string.
    """
    status_secret = os.environ.get("STATUS_SECRET", None)

    if status_secret is not None and get_authorization(request) == status_secret:
        pass
    else:
        authorize(qid, request)

    service: TaggerService = current_app.config["state"]["service"]

    reports = service.status(StatusArgs(
        qid=qid,
        user=None,
        tenant=None,
        title=None
    ))

    return map_all_jobs_status_to_response(reports, status_req)

@tagging_blp.route("/job-status", methods=["GET"])
@tagging_blp.arguments(StatusRequestSchema, location="query")
@tagging_blp.response(200, StatusResponseSchema)
def handle_status(status_req: StatusRequest) -> StatusResponse:
    """Get job statuses for a tenant or user

    Get the status of all jobs for a tenant or user. By default the API returns jobs for the authenticated user. 
    If a tenant id is specified, the API will return jobs for that tenant only if the caller is a tenant admin.
    """
    auth = get_authorization(request)

    service: QueueService = current_app.config["state"]["service"]

    user_info_resolver: UserInfoResolver = current_app.config["state"]["user_info_resolver"]

    args = _get_status_args_and_authorize(status_req, auth, user_info_resolver)

    reports = service.status(args)

    return map_all_jobs_status_to_response(reports, status_req)

def _get_status_args_and_authorize(status_req: StatusRequest, auth: str, user_info_resolver: UserInfoResolver) -> StatusArgs:
    status_req = deepcopy(status_req)
    user_info = user_info_resolver.get_user_info(auth, tenant_id=status_req.tenant)

    if status_req.tenant and not user_info.is_tenant_admin:
        status_req.tenant = None
        status_req.user = user_info.user_adr
    elif status_req.user and not status_req.user == user_info.user_adr:
        raise ForbiddenError(f"Tried to query for user_id={status_req.user} but authenticated user_id={user_info.user_adr}")
    elif not status_req.tenant and not status_req.user:
        # fill in the user
        status_req.user = user_info.user_adr

    args = status_request_to_internal(status_req)

    return args

@tagging_blp.route("/<qid>/stop/<model>", methods=["POST"])
@tagging_blp.response(200, StopTaggingResponseSchema)
def handle_stop_model(qid: str, model: str) -> StopTaggingResponse:
    """Stop tagging jobs by model
    
    Stop a tagging job for a specific model on a given content object (qid).
    """
    q = authorize(qid, request)

    tagger: TaggerService = current_app.config["state"]["service"]

    stop_res = tagger.stop(q.qid, model)

    return map_stop_results_to_response(stop_res)

@tagging_blp.route("/<qid>/stop", methods=["POST"])
@tagging_blp.response(200, StopTaggingResponseSchema)
def handle_stop_content(qid: str) -> StopTaggingResponse:
    """Stop all jobs for a content object
    """
    q = authorize(qid, request)

    tagger: TaggerService = current_app.config["state"]["service"]

    stop_res = tagger.stop(q.qid, None)

    return map_stop_results_to_response(stop_res)

@tagging_blp.route("/models", methods=["GET"])
@tagging_blp.response(200, ListingResponseSchema)
def handle_list_models() -> ListingResponse:
    """List available models"""
    model_configs: dict[str, ModelConfig] = current_app.config["state"]["model_configs"]

    return list_models(model_configs)

@tagging_blp.route("/jobs/<job_id>", methods=["DELETE"])
@tagging_blp.arguments(DeleteJobQuerySchema, location="query")
@tagging_blp.response(204)
def handle_delete_job(args: dict, job_id: str):
    """Delete an inactive job"""
    token = get_authorization(request)

    req = DeleteJobRequest(
        job_id=job_id,
        tenant=args.get("tenant"),
        authorization=token,
    )

    user_info_resolver: UserInfoResolver = current_app.config["state"]["user_info_resolver"]
    js: JobStore = current_app.config["state"]["jobstore"]

    delete_job(req, user_info_resolver=user_info_resolver, js=js)
