from copy import deepcopy
from typing import Any

from cachetools.func import ttl_cache
from requests import HTTPError

from src.common.content import Content, QAPIFactory
from src.common.errors import ExternalServiceError
from src.common.logging import logger
from src.status.get_info import UserInfoResolver

DEFAULTS_TTL = 120
GLOBAL_PROFILES_QID = "iq__3MVS3kjshtnAodRv4qLebBvH3oXb"

class TenantDefaultsResolver:
    """Resolves per-model tagging defaults from named profiles. Reading first from the global default then looking in the
    tenant ml config
    """

    def __init__(self, user_info_resolver: UserInfoResolver, api_factory: QAPIFactory):
        self.user_info_resolver = user_info_resolver
        self.api_factory = api_factory

    def get(self, q: Content, profile: str = "default") -> dict[str, dict[str, Any]]:
        """Returns model name -> job parameters for the given profile."""
        tenant_id = self.user_info_resolver.get_tenant(q.qid, q.token)
        if not tenant_id.startswith("iten"):
            raise ExternalServiceError(f"Unexpected tenant id format: {tenant_id}")

        tenant_profiles = _fetch_tenant_profiles(self.api_factory, tenant_id, q.token)
        global_profiles = _fetch_profiles(self.api_factory, GLOBAL_PROFILES_QID, q.token)

        res = {}
        for key in set(global_profiles) | set(tenant_profiles):
            if not key.startswith("model:"):
                continue
            job = _get_profile(tenant_profiles, key, profile) or _get_profile(global_profiles, key, profile)
            if job is not None:
                res[key[len("model:"):]] = deepcopy(job)
        return res

def _get_profile(profiles: dict[str, Any], key: str, profile: str) -> dict[str, Any] | None:
    model_profiles = profiles.get(key)
    if not isinstance(model_profiles, dict):
        return None
    job = model_profiles.get(profile)
    if job is not None and not isinstance(job, dict):
        logger.warning(f"Invalid profile {profile} for {key}: {job}")
        return None
    return job

@ttl_cache(maxsize=1024, ttl=DEFAULTS_TTL)
def _fetch_tenant_profiles(api_factory: QAPIFactory, tenant_id: str, token: str) -> dict[str, Any]:
    tenant_qid = "iq__" + tenant_id[len("iten"):]
    try:
        tenant_api = api_factory.create(Content(qid=tenant_qid, token=token))
        ml_config_qid = tenant_api.content_object_metadata(metadata_subtree="public/ml_config")
    except HTTPError as e:
        logger.info(f"No ml_config found for tenant {tenant_id}: {e}")
        return {}
    if not isinstance(ml_config_qid, str) or not ml_config_qid:
        logger.warning(f"Invalid ml_config on tenant {tenant_id}: {ml_config_qid}")
        return {}
    return _fetch_profiles(api_factory, ml_config_qid, token)

@ttl_cache(maxsize=1024, ttl=DEFAULTS_TTL)
def _fetch_profiles(api_factory: QAPIFactory, qid: str, token: str) -> dict[str, Any]:
    try:
        profiles = api_factory.create(Content(qid=qid, token=token)).content_object_metadata(metadata_subtree="public/profiles")
    except HTTPError as e:
        logger.info(f"No profiles found on {qid}: {e}")
        return {}
    if not isinstance(profiles, dict):
        logger.warning(f"Invalid profiles on {qid}: {profiles}")
        return {}
    return profiles
