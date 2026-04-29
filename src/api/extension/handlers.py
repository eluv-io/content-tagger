from dataclasses import asdict
from dacite import from_dict
import json

from flask import Response, current_app, request

from src.api.auth import authorize, get_authorization
from src.api_extensions.jobs import DeleteJobRequest
from src.api_extensions.jobs import delete_job
from src.api_extensions.models import list_models
from src.common.errors import BadRequestError
from src.status.get_info import UserInfoResolver
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


def handle_delete_job(job_id: str) -> Response:
    token = get_authorization(request)

    args = request.args.to_dict()

    req = DeleteJobRequest(
        job_id=job_id,
        tenant=args.get("tenant"),
        authorization=token
    )

    user_info_resolver: UserInfoResolver = current_app.config["state"]["user_info_resolver"]
    js: JobStore = current_app.config["state"]["jobstore"]

    delete_job(req, user_info_resolver=user_info_resolver, js=js)

    return Response(status=204)