from dataclasses import asdict
from unittest.mock import Mock, patch

import pytest

from src.api.tagging.handlers import _get_status_args_and_authorize
from src.api.tagging.request_format import StatusRequest
from src.common.errors import ForbiddenError
from src.status.get_info import UserInfo


def _make_resolver(user_adr="0x123", is_tenant_admin=True):
    return Mock(
        get_user_info=Mock(return_value=UserInfo(
            user_adr=user_adr,
            is_tenant_admin=is_tenant_admin,
            is_content_admin=False,
        ))
    )


def test_get_status_args_fills_in_user_when_no_filters():
    """When neither tenant nor user is specified, the authenticated user is filled in."""
    resolver = _make_resolver(user_adr="0x123")
    args = _get_status_args_and_authorize(StatusRequest(), auth="token", user_info_resolver=resolver)
    assert args.user == "0x123"
    assert args.tenant is None


def test_get_status_args_passes_tenant_for_tenant_admin():
    """Tenant admins can filter by tenant."""
    resolver = _make_resolver(user_adr="0x123", is_tenant_admin=True)
    args = _get_status_args_and_authorize(
        StatusRequest(tenant="my_tenant"), auth="token", user_info_resolver=resolver
    )
    assert args.tenant == "my_tenant"


def test_get_status_args_clears_tenant_for_non_admin():
    """Non-admins requesting a tenant filter have it silently cleared and replaced with user."""
    resolver = _make_resolver(user_adr="0x123", is_tenant_admin=False)
    args = _get_status_args_and_authorize(
        StatusRequest(tenant="my_tenant"), auth="token", user_info_resolver=resolver
    )
    assert args.tenant is None
    assert args.user == "0x123"


def test_get_status_args_allows_own_user_filter():
    """A user may filter by their own user address."""
    resolver = _make_resolver(user_adr="0x123")
    args = _get_status_args_and_authorize(
        StatusRequest(user="0x123"), auth="token", user_info_resolver=resolver
    )
    assert args.user == "0x123"


def test_get_status_args_raises_forbidden_for_other_user():
    """Querying for a different user raises ForbiddenError."""
    resolver = _make_resolver(user_adr="0x123")
    with pytest.raises(ForbiddenError):
        _get_status_args_and_authorize(
            StatusRequest(user="0x456"), auth="token", user_info_resolver=resolver
        )

def test_no_args_gives_user():
    resolver = _make_resolver(user_adr="0x123")
    args = _get_status_args_and_authorize(
        StatusRequest(), auth="token", user_info_resolver=resolver
    )
    assert args.user == "0x123"
    assert args.tenant is None


def test_tenant_status(mock_app, make_tag_args):
    client = mock_app.test_client()
    qid = "qid1"

    response = client.post(
        f"/{qid}/tag?authorization=fake", 
        json={
            "options": {
                "destination_qid": "",
                "replace": True,
                "max_fetch_retries": 3,
                "scope": {}
            },
            "jobs": [
                {
                    "model": "test_model",
                    "model_params": {"tags": ["hello1", "hello2"]},
                }
            ]
        }
    )

    assert response.status_code == 200
    response = client.get("/job-status?tenant=test%20tenant&authorization=fake")
    assert response.status_code == 200
    jobs = response.get_json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["tenant"] == "test tenant"