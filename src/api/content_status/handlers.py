from dataclasses import asdict
import json

from flask import Response, current_app, request

from src.api.auth import authorize
from src.status.service import TaggingStatusService


def handle_content_status(qid: str) -> Response:

    q = authorize(qid, request)

    service: TaggingStatusService = current_app.config["state"]["status_service"]

    payload = service.get_content_summary(q=q)

    return Response(
        response=json.dumps(asdict(payload)),
        status=200,
        mimetype="application/json",
    )

def handle_model_status(qid: str, model: str) -> Response:
    q = authorize(qid, request)

    service: TaggingStatusService = current_app.config["state"]["status_service"]

    payload = service.get_model_status(q=q, model=model)

    return Response(
        response=json.dumps(asdict(payload)),
        status=200,
        mimetype="application/json",
    )