from flask import current_app, request
from flask_smorest import Blueprint

from src.api.auth import authorize
from src.status.format import ContentStatusResponse, ContentStatusResponseSchema, ModelStatusResponse, ModelStatusResponseSchema
from src.status.service import TaggingStatusService

content_status_blp = Blueprint(
    "content_status", __name__, description="Per-content tagging status summaries."
)


@content_status_blp.route("/<qid>/tag-status", methods=["GET"])
@content_status_blp.response(200, ContentStatusResponseSchema)
def handle_content_status(qid: str) -> ContentStatusResponse:
    q = authorize(qid, request)

    service: TaggingStatusService = current_app.config["state"]["status_service"]

    return service.get_content_summary(q=q)


@content_status_blp.route("/<qid>/tag-status/<model>", methods=["GET"])
@content_status_blp.response(200, ModelStatusResponseSchema)
def handle_model_status(qid: str, model: str) -> ModelStatusResponse:
    q = authorize(qid, request)

    service: TaggingStatusService = current_app.config["state"]["status_service"]

    return service.get_model_status(q=q, model=model)
