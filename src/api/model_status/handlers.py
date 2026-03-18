from dataclasses import asdict
import json

from flask import Response, current_app

from src.api.tagging.handlers import _get_authorized_content
from src.status.model_status import get_model_status
from src.tagging.fabric_tagging.tagger import FabricTagger


def handle_model_status(qid: str, model: str) -> Response:
    q = _get_authorized_content(qid)

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
