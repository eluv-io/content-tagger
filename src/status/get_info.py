
from dataclasses import dataclass

import requests
from loguru import logger
from functools import lru_cache

from src.common.errors import ExternalServiceError
from src.tags.tagstore.abstract import Tagstore

@dataclass(frozen=True)
class UserInfo: 
    user_adr: str
    is_tenant_admin: bool
    is_content_admin: bool

@dataclass(frozen=True)
class UserInfoResolverConfig:
    fabric_url: str
    user_info_url: str

class UserInfoResolver:
    def __init__(
        self, 
        cfg: UserInfoResolverConfig
    ):
        self.fabric_url = cfg.fabric_url
        self.user_info_url = cfg.user_info_url

    @lru_cache(maxsize=128)
    def get_tenant(self, qid: str, token: str) -> str:
        """Resolve the tenant ID for a given content object ID using the fabric profile endpoint."""
        resp = requests.get(
            f"{self.fabric_url}/q/{qid}?profile&authorization={token}"
        ).json()
        if "content_profile" not in resp or "tenant_id" not in resp["content_profile"]:
            raise ExternalServiceError("Failed to get tenant ID from fabric profile endpoint")
        return resp["content_profile"]["tenant_id"]

    @lru_cache(maxsize=128)
    def get_user_info(self, auth: str, tenant_id: str | None) -> UserInfo:
        """Get user info for a given tenant ID using the user info endpoint."""
        resp = requests.get(self.user_info_url, params={"tenant": tenant_id, "authorization": auth})
        resp.raise_for_status()
        data = resp.json()

        is_content_admin, is_tenant_admin = False, False
        if "token_data" not in data or "adr" not in data["token_data"]:
            raise ExternalServiceError("Failed to get user address from token_info service")
        if tenant_id is not None and ("is_tenant_admin" not in data or "is_content_admin" not in data):
            raise ExternalServiceError("Failed to get admin status from token_info service")
        elif tenant_id is not None:
            is_content_admin = data["is_tenant_admin"]
            is_tenant_admin = data["is_content_admin"]
        return UserInfo(
            user_adr=data["token_data"]["adr"],
            is_tenant_admin=is_tenant_admin,
            is_content_admin=is_content_admin
        )