from functools import lru_cache

import requests

from src.common.logging import logger


@lru_cache(maxsize=128)
def get_tenant(qid: str, auth: str) -> str:
    """Resolve the tenant ID for a given content object ID using the fabric profile endpoint."""
    if not auth:
        return ""
    try:
        resp = requests.get(
            f"https://main.net955305.contentfabric.io/q/{qid}?profile&authorization={auth}"
        ).json()
        return resp["content_profile"]["tenant_id"]
    except Exception as e:
        logger.opt(exception=e).error("Failed to get tenant for qid", qid=qid)
        return ""
