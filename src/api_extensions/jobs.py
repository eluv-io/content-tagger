from dataclasses import dataclass
from loguru import logger

from src.common.errors import BadRequestError, ForbiddenError
from src.status.get_info import UserInfoResolver
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.model import UpdateJobRequest

@dataclass
class DeleteJobRequest:
    job_id: str
    tenant: str | None
    authorization: str


def delete_job(
    req: DeleteJobRequest,
    user_info_resolver: UserInfoResolver,
    js: JobStore
) -> None:
    user_info = user_info_resolver.get_user_info(
        auth=req.authorization,
        tenant=req.tenant
    )
    if req.tenant and not user_info.is_tenant_admin:
        raise ForbiddenError("Only tenant admins can query by tenant")
    
    item = js.get_job(req.job_id)

    if item.status not in ("succeeded", "failed", "cancelled"):
        raise BadRequestError(f"Only jobs with status 'succeeded', 'failed' or 'cancelled' can be deleted.")

    if not user_info.is_tenant_admin and user_info.user_adr != item.user:
        raise ForbiddenError(
            f"Tried to delete job for user_id={item.user} but authenticated user_id={user_info.user_adr}. Please specify a tenant id if you are a tenant admin."
        )

    js.update_job(UpdateJobRequest(id=req.job_id, status="deleted"), req.authorization)

    logger.info(f"Deleted job {req.job_id} for user {item.user} and tenant {item.tenant}")