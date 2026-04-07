from dataclasses import asdict
import json

from flask import Response, current_app

from src.api_extensions.models import list_models
from src.tag_containers.registry import ContainerRegistry
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
