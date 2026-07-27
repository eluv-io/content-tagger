from flask import current_app, request
from flask_smorest import Blueprint

from src.api.auth import get_authorization
from src.api_extensions.jobs import DeleteJobQuerySchema, DeleteJobRequest, delete_job
from src.api_extensions.models import ListingResponse, ListingResponseSchema, list_models
from src.common.model import ModelConfig
from src.status.get_info import UserInfoResolver
from src.tagging.fabric_tagging.queue.abstract import JobStore

extension_blp = Blueprint(
    "extension", __name__, description="Model listing and job management."
)


@extension_blp.route("/models", methods=["GET"])
@extension_blp.response(200, ListingResponseSchema)
def handle_list_models() -> ListingResponse:
    """List available models"""
    model_configs: dict[str, ModelConfig] = current_app.config["state"]["model_configs"]

    return list_models(model_configs)


@extension_blp.route("/jobs/<job_id>", methods=["DELETE"])
@extension_blp.arguments(DeleteJobQuerySchema, location="query")
@extension_blp.response(204)
def handle_delete_job(args: dict, job_id: str):
    """Delete an inactive job"""
    token = get_authorization(request)

    req = DeleteJobRequest(
        job_id=job_id,
        tenant=args.get("tenant"),
        authorization=token,
    )

    user_info_resolver: UserInfoResolver = current_app.config["state"]["user_info_resolver"]
    js: JobStore = current_app.config["state"]["jobstore"]

    delete_job(req, user_info_resolver=user_info_resolver, js=js)
