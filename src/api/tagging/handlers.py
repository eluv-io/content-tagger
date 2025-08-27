import json

from flask import Response, request, current_app

from src.api.tagging.format import TagAPIArgs, ImageTagAPIArgs
from src.common.errors import BadRequestError
from src.api.auth import get_authorization
from src.common.content import Content
from src.tagger.fabric_tagging.tagger import FabricTagger

def handle_tag(qhit: str) -> Response:
    auth = get_authorization(request)

    q = Content(qhit, auth)

    try:
        args = TagAPIArgs.from_dict(request.json)
    except BadRequestError as e:
        raise e
    except Exception as e:
        raise BadRequestError(
            "Invalid search arguments. Please check your query parameters") from e

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    status = tagger.tag(q, args.to_tag_args())

    return Response(response=json.dumps(status), status=200, mimetype='application/json')

def handle_image_tag(qhit: str) -> Response:
    auth = get_authorization(request)

    q = Content(qhit, auth)

    try:
        args = ImageTagAPIArgs.from_dict(request.json)
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
    auth = get_authorization(request)

    q = Content(qhit, auth)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    res = tagger.status(q.qhit)

    return Response(response=json.dumps(res), status=200, mimetype='application/json')


def handle_stop(
        qhit: str, 
        feature: str
    ) -> Response:
    auth = get_authorization(request)

    q = Content(qhit, auth)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    tagger.stop(q.qhit, feature)

    return Response(response=json.dumps({'message': f"Stopping {feature} on {qhit}. Check with /status for completion."}), status=200, mimetype='application/json')