import json

from flask import Response, request, current_app
from src.common.logging import logger

from src.api.tagging.format import TagAPIArgs, ImageTagAPIArgs
from src.common.errors import BadRequestError
from src.api.auth import get_authorization
from src.common.content import Content, ContentFactory
from src.tagging.fabric_tagging.tagger import FabricTagger

def handle_tag(qhit: str) -> Response:
    q = _get_authorized_content(qhit)

    try:
        body = request.json
        if body is None:
            raise BadRequestError("Missing request body")
        args = TagAPIArgs.from_dict(body)
    except BadRequestError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise BadRequestError(
            "Invalid arguments. Please check your request body.") from e

    logger.debug(args)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    status = tagger.tag(q, args.to_tag_args())

    return Response(response=json.dumps(status), status=200, mimetype='application/json')

def handle_image_tag(qhit: str) -> Response:
    q = _get_authorized_content(qhit)

    try:
        body = request.json
        if body is None:
            raise BadRequestError("Missing request body")
        args = ImageTagAPIArgs.from_dict(body)
    except TypeError as e:
        return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')

    # TODO: belongs in FabricTagger
    # for feature, run_config in args.features.items():
    #     if not config["services"][feature].get("frame_level", False):
    #         return Response(response=json.dumps({'error': f"Image tagging for {feature} is not supported"}), status=400, mimetype='application/json')

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    status = tagger.tag(q, args.to_tag_args())

    return Response(response=json.dumps(status), status=200, mimetype='application/json')


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