
import os
import pytest

from src.common.content import Content
from src.fetch.factory import FetchFactory
from src.fetch.model import FetcherConfig

@pytest.fixture
def fetcher(fetcher_config: FetcherConfig, tag_store, qfactory) -> FetchFactory:
    """Create a FetchFactory instance for testing"""
    return FetchFactory(config=fetcher_config, ts=tag_store, qfactory=qfactory)

@pytest.fixture
def legacy_vod_content_with_tags_clean(q_legacy, tag_store) -> Content:

    batch_ids = tag_store.find_batches(q=q_legacy)
    for batch_id in batch_ids:
        tag_store.delete_batch(batch_id, q=q_legacy)

    return q_legacy


@pytest.fixture
def vod_content_with_tags_clean(q, tag_store) -> Content:

    batch_ids = tag_store.find_batches(q=q)
    for batch_id in batch_ids:
        tag_store.delete_batch(batch_id, q=q)

    return q

@pytest.fixture
def assets_content_with_tags_clean(q_assets, tag_store) -> Content:
    batch_ids = tag_store.find_batches(q=q_assets)
    for batch_id in batch_ids:
        tag_store.delete_batch(batch_id, q=q_assets)

    return q_assets