from copy import deepcopy
from typing import Any

from cachetools.func import ttl_cache
from requests import HTTPError

from src.common.content import Content, QAPIFactory
from src.common.errors import ExternalServiceError
from src.common.logging import logger
from src.status.get_info import UserInfoResolver

DEFAULTS_TTL = 120

class TenantDefaultsResolver:
    """Fetches per-model tagging defaults from the tenant's ml config object.

    Results are cached for DEFAULTS_TTL seconds, so edits to a tenant's defaults take up to that long to apply.
    """

    def __init__(self, user_info_resolver: UserInfoResolver, api_factory: QAPIFactory):
        self.user_info_resolver = user_info_resolver
        self.api_factory = api_factory

    def get(self, q: Content) -> dict[str, dict[str, Any]]:
        tenant_id = self.user_info_resolver.get_tenant(q.qid, q.token)
        if not tenant_id.startswith("iten"):
            raise ExternalServiceError(f"Unexpected tenant id format: {tenant_id}")
        return deepcopy(_fetch_defaults(self.api_factory, tenant_id, q.token))

@ttl_cache(maxsize=1024, ttl=DEFAULTS_TTL)
def _fetch_defaults(api_factory: QAPIFactory, tenant_id: str, token: str) -> dict[str, dict[str, Any]]:
    tenant_qid = "iq__" + tenant_id[len("iten"):]

    try:
        tenant_api = api_factory.create(Content(qid=tenant_qid, token=token))
        ml_config_qid = tenant_api.content_object_metadata(metadata_subtree="public/ml_config")
        if not isinstance(ml_config_qid, str) or not ml_config_qid:
            logger.warning(f"Invalid ml_config on tenant {tenant_id}: {ml_config_qid}")
            return {}
        ml_config_api = api_factory.create(Content(qid=ml_config_qid, token=token))
        defaults = ml_config_api.content_object_metadata(metadata_subtree="public/tagging/model_defaults")
    except HTTPError as e:
        logger.info(f"No tenant model defaults found for tenant {tenant_id}: {e}")
        return {}

    if not isinstance(defaults, dict):
        logger.warning(f"Invalid model_defaults for tenant {tenant_id}: {defaults}")
        return {}

    return defaults
