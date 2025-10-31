import json

from flask import Response, request, current_app
from dacite import from_dict

from src.api.tagging.dto_mapping import tag_args_from_req
from src.common.logging import logger

from src.api.tagging.format import ImageTagAPIArgs
from src.common.errors import BadRequestError
from src.api.auth import *
from src.common.content import Content, ContentFactory
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.api.tagging.dto_mapping import *

def handle_tag(qhit: str) -> Response:
    q = _get_authorized_content(qhit)
    args = tag_args_from_req(q)
    logger.debug(args)

    _validate_destination_auth(q, args.destination_qid)
    
    tagger: FabricTagger = current_app.config["state"]["tagger"]
    tag_args = map_video_tag_dto(args, tagger.cregistry, q)
    
    return _execute_tagging(q, tag_args)

def handle_image_tag(qhit: str) -> Response:
    q = _get_authorized_content(qhit)
    
    try:
        body = request.json
        assert body is not None
        args = from_dict(ImageTagAPIArgs, body)
    except Exception as e:
        raise BadRequestError(f"Invalid request body: {e}") from e

    _validate_destination_auth(q, args.destination_qid)

    tag_args = map_asset_tag_dto(args)
    
    return _execute_tagging(q, tag_args)

def _execute_tagging(q: Content, tag_args: list[TagArgs]) -> Response:
    """Execute tagging for multiple features and return status response.
    
    Handles destination authorization if specified in any tag_arg.
    """
    tagger: FabricTagger = current_app.config["state"]["tagger"]
    
    status_by_feature: dict[str, str] = {}
    for tag_arg in tag_args:
        status_by_feature[tag_arg.feature] = tagger.tag(q, tag_arg)
    
    return Response(
        response=json.dumps(status_by_feature), 
        status=200, 
        mimetype='application/json'
    )

def handle_status(qhit: str) -> Response:
    q = _get_authorized_content(qhit)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    res = tagger.status(q.qhit)

    return Response(response=json.dumps(res), status=200, mimetype='application/json')

def handle_stop(
        qhit: str, 
        feature: str
    ) -> Response:
    q = _get_authorized_content(qhit)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    tagger.stop(q.qhit, feature, None)

    return Response(response=json.dumps({'message': f"Stopping {feature} on {qhit}. Check with /status for completion."}), status=200, mimetype='application/json')

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