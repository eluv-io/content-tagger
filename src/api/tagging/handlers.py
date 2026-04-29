from copy import deepcopy
from dataclasses import asdict
from dacite import from_dict, Config
import json
import os
from common_ml.utils.metrics import timeit

from flask import Response, request, current_app
from dacite import from_dict

from src.api.arg_resolver import ArgsResolver
from src.service.abstract import TaggerService
from src.api.tagging.response_format import StartStatus, StartTaggingResponse
from src.common.logging import logger

from src.common.errors import *
from src.api.auth import *
from src.common.content import Content
from src.api.tagging.request_mapping import *
from src.api.tagging.response_mapping import *
from src.service.impl.queue_based import QueueService
from src.status.get_info import UserInfoResolver

def handle_tag(qid: str) -> Response:
    q = authorize(qid, request)

    try:
        args = from_dict(data_class=StartJobsRequest, data=request.get_json(), config=Config(strict=True))
    except Exception as e:
        raise BadRequestError(f"Invalid request body format: {e}") from e
    
    logger.debug(args)

    if args.options.destination_qid:
        authorize(args.options.destination_qid, request)

    arg_resolver: ArgsResolver = current_app.config["state"]["arg_resolver"]

    with timeit("resolving tag args"):
        tag_args = arg_resolver.resolve(args, q)

    return _execute_tagging(q, tag_args)

def _execute_tagging(q: Content, tag_args: list[TagArgs]) -> Response:
    """Execute tagging for multiple features and return start status response."""
    tagger: TaggerService = current_app.config["state"]["service"]
    
    jobs: list[StartStatus] = []
    for tag_arg in tag_args:
        try:
            with timeit(f"tagging for feature {tag_arg.feature}"):
                result = tagger.tag(q, tag_arg)
        except Exception as e:
            logger.opt(exception=e).error("Failed to start tagging", feature=tag_arg.feature, qid=q.qid)
            jobs.append(
                StartStatus(
                    job_id="",
                    model=tag_arg.feature,
                    started=False,
                    message="Tag job failed to start",
                    error=str(e),
                )
            )
            continue

        jobs.append(
            StartStatus(
                job_id=result.job_id,
                model=tag_arg.feature,
                started=result.started,
                message=result.message,
                error=None,
            )
        )

    payload = StartTaggingResponse(jobs=jobs)

    return Response(
        response=json.dumps(asdict(payload)),
        status=200,
        mimetype="application/json",
    )

def handle_status_content(qid: str) -> Response:
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

    status_req = _parse_status_request()

    response = map_all_jobs_status_to_response(reports, status_req)

    return Response(response=json.dumps(asdict(response)), status=200, mimetype='application/json')

def handle_status() -> Response:
    """Global job-status endpoint. Requires ?tenant= filter.
    
    Authentication: the caller's auth token is verified by picking the first
    returned job's qid and confirming get_tenant(qid, auth) matches the
    requested tenant.
    """
    auth = get_authorization(request)

    status_req = _parse_status_request()

    service: QueueService = current_app.config["state"]["service"]
    
    user_info_resolver: UserInfoResolver = current_app.config["state"]["user_info_resolver"]

    args = _get_status_args_and_authorize(status_req, auth, user_info_resolver)

    reports = service.status(args)

    response = map_all_jobs_status_to_response(reports, status_req)

    return Response(response=json.dumps(asdict(response)), status=200, mimetype='application/json')

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

def _parse_status_request() -> StatusRequest:
    """Parse status query parameters into a StatusRequest."""
    try:
        args = request.args.to_dict()
        if "authorization" in args:
            del args["authorization"]
        return from_dict(data_class=StatusRequest, data=args, config=Config(strict=True, cast=[int]))
    except Exception as e:
        raise BadRequestError(f"Invalid status query parameters: {e}") from e

def handle_stop_model(
    qid: str, 
    feature: str
) -> Response:
    q = authorize(qid, request)

    tagger: TaggerService = current_app.config["state"]["service"]

    stop_res = tagger.stop(q.qid, feature)

    api_res = map_stop_results_to_response(stop_res)

    return Response(response=json.dumps(asdict(api_res)), status=200, mimetype='application/json')

def handle_stop_content(
    qid: str
) -> Response:
    q = authorize(qid, request)

    tagger: TaggerService = current_app.config["state"]["service"]

    stop_res = tagger.stop(q.qid, None)

    api_res = map_stop_results_to_response(stop_res)

    return Response(response=json.dumps(asdict(api_res)), status=200, mimetype='application/json')