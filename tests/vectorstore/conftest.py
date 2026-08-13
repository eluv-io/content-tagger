
import os
import uuid

import pytest
import requests

from src.common.content import Content
from src.tags.datastore.model import Batch, Tag
from src.tags.vectorstore.rest_vectorstore import RestVectorstore

# sources these tests write under
SOURCES = ["test_source_1", "test_source_2"]

# only used when the index has to be created; an existing index keeps its own size.
# the service only accepts 256, 512 or 1024
DEFAULT_VECTOR_SIZE = 256


@pytest.fixture
def vectorstore_host() -> str:
    """Base url of a live vectorstore. Without it there is nothing to test against."""
    host = os.getenv("TEST_VECTORSTORE_HOST")
    if not host:
        pytest.skip("TEST_VECTORSTORE_HOST not set in environment")
    if not os.getenv("TEST_AUTH"):
        pytest.skip("TEST_AUTH not set in environment")
    return host


@pytest.fixture
def index_qid() -> str:
    return "iq__4V1Vr9QefogNywPAMCU1fhZxcFF5"


@pytest.fixture
def vector_api(vectorstore_host: str, index_qid: str, q: Content) -> "VectorApi":
    """Raw client for the endpoints the Datastore protocol does not expose."""
    return VectorApi(vectorstore_host, index_qid, q)


class VectorApi:
    def __init__(self, base_url: str, index_qid: str, q: Content):
        self.base = f"{base_url.rstrip('/')}/indexes/{index_qid}"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {q.token}"}

    def get_index(self) -> dict | None:
        res = requests.get(self.base, headers=self.headers, timeout=10)
        if res.status_code == 404:
            return None
        res.raise_for_status()
        return res.json()

    def create_index(self, vector_size: int) -> None:
        res = requests.post(
            self.base,
            json={"name": "content-tagger integration tests", "vector_size": vector_size},
            headers=self.headers,
            timeout=10,
        )
        res.raise_for_status()

    def tracks(self) -> dict[str, int]:
        res = requests.get(f"{self.base}/tracks", headers=self.headers, timeout=10)
        res.raise_for_status()
        return {t["name"]: t["count"] for t in res.json().get("tracks", [])}

    def search(self, vector: list[float], track: str | None = None, limit: int = 20) -> list[dict]:
        body: dict = {"vector": vector, "limit": limit}
        if track is not None:
            body["track"] = track
        res = requests.post(f"{self.base}/search", json=body, headers=self.headers, timeout=10)
        res.raise_for_status()
        return [r["vector"] for r in res.json().get("results", [])]


@pytest.fixture
def vector_index(vector_api: VectorApi) -> dict:
    """The index these tests write into, created on first use."""
    index = vector_api.get_index()
    if index is None:
        vector_api.create_index(vector_size=DEFAULT_VECTOR_SIZE)
        index = vector_api.get_index()
        assert index is not None, "index was created but could not be read back"
    return index


@pytest.fixture
def vector_size(vector_index: dict) -> int:
    """Vectors must match the index's configured dimension."""
    size = vector_index.get("vector_size")
    if not size:
        pytest.skip("index does not report a vector_size")
    return size


@pytest.fixture
def vector_model() -> str:
    """A unique model per test, so a test can never see another run's vectors."""
    return f"tagger_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def live_vectorstore(vectorstore_host, index_qid, vector_index, q):
    """A store that records every batch it creates and deletes them all on teardown,
    so a test never has to remember to clean up after itself.

    Depends on vector_index so that the index exists before the first write.
    """
    store = RestVectorstore(base_url=vectorstore_host, timeout=10, index_qid=index_qid)

    created: list[str] = []
    create_batch = store.create_batch

    def recording_create_batch(model: str, author: str, q: Content) -> Batch:
        batch = create_batch(model=model, author=author, q=q)
        created.append(batch.id)
        return batch

    store.create_batch = recording_create_batch  # type: ignore[method-assign]

    yield store

    for batch_id in created:
        try:
            store.delete_batch(batch_id, q=q)
        except requests.exceptions.HTTPError as e:
            # a test that deleted its own batch is fine, anything else is not: a vector
            # left with no batch can no longer be deleted at all
            if e.response is None or e.response.status_code != 404:
                raise


@pytest.fixture
def make_vector(vector_size: int):
    """Build a vector of the index's dimension. Each seed picks its own dimension, so
    vectors from different seeds are orthogonal and a search cannot confuse them.

    `dims` overrides the length, for the cases where a model's output does not match
    what the index expects."""
    def fn(seed: float = 1.0, source: str = SOURCES[0], start_time: int = 0,
           end_time: int = 1000, frame_info=None, batch_id: str = "",
           dims: int | None = None) -> Tag:
        dims = vector_size if dims is None else dims
        data = [0.0] * dims
        if dims:
            data[int(seed) % dims] = 1.0
        return Tag(
            id="",
            start_time=start_time,
            end_time=end_time,
            data=data,
            additional_info={"confidence": 0.9},
            source=source,
            batch_id=batch_id,
            frame_info=frame_info,
        )
    return fn
