from flask import current_app, request
from flask_smorest import Blueprint

from src.api.auth import authorize
from src.status.format import ContentStatusResponse, ContentStatusResponseSchema, ModelStatusResponse, ModelStatusResponseSchema
from src.status.service import TaggingStatusService

content_status_blp = Blueprint(
    "Tagging History", __name__, description="Per-content tagging status summaries."
)


@content_status_blp.route("/<qid>/tag-status", methods=["GET"])
@content_status_blp.response(200, ContentStatusResponseSchema)
def handle_content_status(qid: str) -> ContentStatusResponse:
    """Get tagging status summary for a content object"""
    q = authorize(qid, request)

    service: TaggingStatusService = current_app.config["state"]["status_service"]

    return service.get_content_summary(q=q)


@content_status_blp.route("/<qid>/tag-status/<model>", methods=["GET"])
@content_status_blp.response(200, ModelStatusResponseSchema)
def handle_model_status(qid: str, model: str) -> ModelStatusResponse:
    """Get tagging status for a specific model on a content object"""
    q = authorize(qid, request)

    service: TaggingStatusService = current_app.config["state"]["status_service"]

    return service.get_model_status(q=q, model=model)
