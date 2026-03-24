from dataclasses import asdict
import json

from flask import Response, current_app, request

from src.api.auth import authorize
from src.status.content_status import get_content_summary
from src.tagging.fabric_tagging.tagger import TaggerWorker


def handle_content_status(qid: str) -> Response:

    q = authorize(qid, request)

    tagger: TaggerWorker = current_app.config["state"]["tagger"]

    payload = get_content_summary(
        q=q,
        tagstore=tagger.tagstore,
        track_resolver=tagger.track_resolver,
    )

    return Response(
        response=json.dumps(asdict(payload)),
        status=200,
        mimetype="application/json",
    )
