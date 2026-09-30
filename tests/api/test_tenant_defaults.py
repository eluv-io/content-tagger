from unittest.mock import Mock

import pytest
from requests import HTTPError

from src.api import tenant_defaults
from src.api.tenant_defaults import GLOBAL_PROFILES_QID, TenantDefaultsResolver
from src.common.content import Content

TENANT_ID = "iten123"
TENANT_QID = "iq__123"
ML_CONFIG_QID = "iq__mlconfig"
CONTENT = Content(qid="iq__content", token="token")

GLOBAL_PROFILES = {
    "model:music": {
        "default": {"model_params": {"ground_truth": "default.pklz"}},
        "loose": {"model_params": {"ground_truth": "default.pklz", "min_count": 7}},
    },
    "model:asr": {
        "default": {"model_params": {"word_level": False}},
    },
    "not_a_model": {
        "default": {"model_params": {"ignored": True}},
    },
}

TENANT_PROFILES = {
    "model:asr": {
        "default": {"model_params": {"prettify": True, "word_level": True}},
        "spanish": {
            "model_params": {"prettify": True, "word_level": True},
            "overrides": {"scope": {"stream": "spanish_audio_5_1"}},
            "track_suffix": "spanish",
        },
    },
}

def content_object_metadata(qid: str, metadata_subtree: str):
    if qid == TENANT_QID and metadata_subtree == "public/ml_config":
        return ML_CONFIG_QID
    if qid == ML_CONFIG_QID and metadata_subtree == "public/profiles":
        return TENANT_PROFILES
    if qid == GLOBAL_PROFILES_QID and metadata_subtree == "public/profiles":
        return GLOBAL_PROFILES
    raise HTTPError(f"{qid}/meta/{metadata_subtree} not found")

@pytest.fixture
def resolver():
    tenant_defaults._fetch_profiles.cache_clear()
    tenant_defaults._fetch_tenant_profiles.cache_clear()

    user_info = Mock()
    user_info.get_tenant.return_value = TENANT_ID

    api_factory = Mock()
    api_factory.create.side_effect = lambda q: Mock(
        content_object_metadata=lambda metadata_subtree: content_object_metadata(q.qid, metadata_subtree)
    )

    return TenantDefaultsResolver(user_info, api_factory)

def test_default_profile(resolver):
    assert resolver.get(CONTENT) == {
        # tenant profile beats global
        "asr": TENANT_PROFILES["model:asr"]["default"],
        # tenant doesn't define music, so falls back to global
        "music": GLOBAL_PROFILES["model:music"]["default"],
    }

def test_named_profile(resolver):
    assert resolver.get(CONTENT, profile="spanish") == {"asr": TENANT_PROFILES["model:asr"]["spanish"]}
    assert resolver.get(CONTENT, profile="loose") == {"music": GLOBAL_PROFILES["model:music"]["loose"]}

def test_tenant_without_ml_config(resolver):
    resolver.user_info_resolver.get_tenant.return_value = "itenNoMlConfig"

    assert resolver.get(CONTENT) == {
        "asr": GLOBAL_PROFILES["model:asr"]["default"],
        "music": GLOBAL_PROFILES["model:music"]["default"],
    }

def test_unreadable_global(resolver, monkeypatch):
    monkeypatch.setattr(tenant_defaults, "GLOBAL_PROFILES_QID", "iq__unreadable")

    assert resolver.get(CONTENT) == {"asr": TENANT_PROFILES["model:asr"]["default"]}
