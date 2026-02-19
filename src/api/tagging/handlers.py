from dataclasses import asdict
import json
import os

from flask import Response, request, current_app
from dacite import from_dict

from src.api.tagging.request_mapping import map_video_tag_dto
from src.api.tagging.response_format import StartStatus, StartTaggingResponse
from src.api.tagging.response_format import StartStatus
from src.common.logging import logger

from src.common.errors import *
from src.api.auth import *
from src.common.content import Content, ContentFactory
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.api.tagging.request_mapping import *
from src.api.tagging.response_mapping import *

def handle_tag(qhit: str) -> Response:
    q = _get_authorized_content(qhit)

    try:
        args = from_dict(data_class=StartJobsRequest, data=request.get_json(), config=Config(strict=True))
    except Exception as e:
        raise BadRequestError(f"Invalid request body format: {e}") from e
    
    logger.debug(args)

    _validate_destination_auth(q, args.defaults.destination_qid)
    
    tagger: FabricTagger = current_app.config["state"]["tagger"]
    tag_args = map_video_tag_dto(args, tagger.cregistry, q)
    
    return _execute_tagging(q, tag_args)


def _execute_tagging(q: Content, tag_args: list[TagArgs]) -> Response:
    """Execute tagging for multiple features and return start status response."""
    tagger: FabricTagger = current_app.config["state"]["tagger"]
    
    jobs: list[StartStatus] = []
    for tag_arg in tag_args:
        try:
            result = tagger.tag(q, tag_arg)
        except Exception as e:
            logger.opt(exception=e).error("Failed to start tagging", feature=tag_arg.feature, qhit=q.qhit)
            jobs.append(
                StartStatus(
                    job_id="",
                    model=tag_arg.feature,
                    stream="",
                    started=False,
                    message="Tag job failed to start",
                    error=str(e),
                )
            )
            continue

        job_id = result.job_id
        jobs.append(
            StartStatus(
                job_id=str(job_id),
                model=job_id.feature,
                stream=job_id.stream,
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

def handle_status(qhit: str) -> Response:
    status_secret = os.environ.get("STATUS_SECRET", None)
    
    if status_secret is not None and get_authorization(request) == status_secret:
        pass
    else:
        _get_authorized_content(qhit)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    reports = tagger.status(qhit)

    return Response(response=json.dumps(asdict(map_all_jobs_status_to_response(reports))), status=200, mimetype='application/json')

def handle_stop_model(
    qhit: str, 
    feature: str
) -> Response:
    q = _get_authorized_content(qhit)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    stop_res = tagger.stop(q.qhit, feature, None)

    api_res = map_stop_results_to_response(stop_res)

    return Response(response=json.dumps(asdict(api_res)), status=200, mimetype='application/json')

def handle_stop_content(
    qhit: str
) -> Response:
    q = _get_authorized_content(qhit)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    stop_res = tagger.stop(q.qhit, None, None)

    api_res = map_stop_results_to_response(stop_res)

    return Response(response=json.dumps(asdict(api_res)), status=200, mimetype='application/json')

def _get_authorized_content(qhit: str) -> Content:
    auth = get_authorization(request)
    qfactory: ContentFactory = current_app.config["state"]["content_factory"]
    return qfactory.create_content(qhit, auth)

def _validate_destination_auth(source_q: Content, dest_qid: str) -> None:
    """Validate that the destination qid is accessible with the same auth context."""
    if not dest_qid:
        return
    if not is_same_auth_ctx(source_q, dest_qid):
        raise BadRequestError(
            f"Destination content {dest_qid} and source content {source_q.qid} do not share the same authorization context."
        )