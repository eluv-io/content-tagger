
from dataclasses import dataclass

import requests
from loguru import logger
from functools import lru_cache

from src.common.content import Content
from src.common.errors import ExternalServiceError
from src.tags.tagstore.abstract import Tagstore

@dataclass(frozen=True)
class UserInfo: 
    user_adr: str
    is_tenant_admin: bool
    is_content_admin: bool

class UserInfoResolver:
    def __init__(
        self, 
        fabric_url: str, 
        user_info_url: str
    ):
        self.fabric_url = fabric_url
        self.user_info_url = user_info_url

    @lru_cache(maxsize=128)
    def get_tenant(self, q: Content) -> str:
        """Resolve the tenant ID for a given content object ID using the fabric profile endpoint."""
        if not q.token:
            return ""
        resp = requests.get(
            f"{self.fabric_url}/q/{q.qid}?profile&authorization={q.token}"
        ).json()
        if "content_profile" not in resp or "tenant_id" not in resp["content_profile"]:
            raise ExternalServiceError("Failed to get tenant ID from fabric profile endpoint")
        return resp["content_profile"]["tenant_id"]

    @lru_cache(maxsize=128)
    def get_user_info(self, auth: str, tenant_id: str | None) -> UserInfo:
        """Get user info for a given tenant ID using the user info endpoint."""
        resp = requests.get(self.user_info_url, params={"tenant_id": tenant_id}).json()
        is_content_admin, is_tenant_admin = False, False
        if "token_data" not in resp or "adr" not in resp["token_data"]:
            raise ExternalServiceError("Failed to get user address from token_info service")
        if tenant_id is not None and "is_tenant_admin" not in resp or "is_content_admin" not in resp:
            raise ExternalServiceError("Failed to get admin status from token_info service")
        else:
            is_content_admin = resp["is_tenant_admin"]
            is_tenant_admin = resp["is_content_admin"]
        return UserInfo(
            user_adr=resp["token_data"]["adr"],
            is_tenant_admin=is_tenant_admin,
            is_content_admin=is_content_admin
        )