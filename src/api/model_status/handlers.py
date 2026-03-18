from dataclasses import asdict
import json

from flask import Response, current_app, request

from src.api.auth import authorize
from src.status.model_status import get_model_status
from src.tagging.fabric_tagging.tagger import FabricTagger


def handle_model_status(qid: str, model: str) -> Response:
    q = authorize(qid, request)

    tagger: FabricTagger = current_app.config["state"]["tagger"]

    payload = get_model_status(
        q=q,
        model=model,
        tagstore=tagger.tagstore,
        track_resolver=tagger.track_resolver,
    )

    return Response(
        response=json.dumps(asdict(payload)),
        status=200,
        mimetype="application/json",
    )
