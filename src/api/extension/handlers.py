from dataclasses import asdict
import json

from flask import Response, current_app, request

from src.api.auth import authorize
from src.api_extensions.jobs import delete_job
from src.api_extensions.models import list_models
from src.tag_containers.registry import ContainerRegistry
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tags.track_resolver import TrackResolver


def handle_list_models() -> Response:
    registry: ContainerRegistry = current_app.config["state"]["container_registry"]
    track_resolver: TrackResolver = current_app.config["state"]["track_resolver"]

    payload = list_models(registry, track_resolver)

    return Response(
        response=json.dumps(asdict(payload)),
        status=200,
        mimetype="application/json",
    )


def handle_delete_job(qid: str, job_id: str) -> Response:
    q = authorize(qid, request)
    js: JobStore = current_app.config["state"]["jobstore"]

    delete_job(job_id, auth=q.token, js=js)

    return Response(status=204)
