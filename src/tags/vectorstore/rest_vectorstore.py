import json
import time
from typing import cast

import requests

from src.common.content import Content
from src.common.logging import logger
from src.tags.datastore.abstract import Datastore
from src.tags.datastore.model import Batch, Tag, Track, is_vector

class RestVectorstore(Datastore):
    """Datastore backed by the elv-vectorstore service, bound to a single index.

    `index_qid` is the content that holds the index; the `q` passed to each method is
    the content being embedded, which is what a vector's `qid` refers to.

    The service writes and deletes vectors but never lists them, so the read side of
    the protocol (find_tags, count_tags, get_batch, find_batches) is unsupported.

    Every vector written is fitted to the index's configured `vector_size`: a model that
    emits a shorter embedding is zero padded, a longer one is refused.
    """

    def __init__(self, base_url: str, timeout: int, index_qid: str):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.index_qid = index_qid
        self.session = requests.Session()
        self._vector_size: int | None = None

    def _index_url(self, path: str = "") -> str:
        return f"{self.base_url}/indexes/{self.index_qid}{path}"

    def _get_headers(self, q: Content) -> dict:
        return {'Content-Type': 'application/json', 'Authorization': f"Bearer {q.token}"}

    def _log_response_and_raise(self, response: requests.Response):
        try:
            logger.error(f"{json.dumps(response.json())}")
        except Exception:
            logger.error(f"HTTP {response.status_code} response (non-JSON): {response.text}")
        response.raise_for_status()

    def _get_vector_size(self, q: Content) -> int:
        """Dimension the index is configured for, read once and cached for the store's life.
        """
        if self._vector_size is None:
            response = self.session.get(
                self._index_url(),
                headers=self._get_headers(q),
                timeout=self.timeout
            )

            if not response.ok:
                self._log_response_and_raise(response)

            self._vector_size = int(response.json()["vector_size"])

        return self._vector_size

    def _fit_vector(self, vector: list[float], size: int) -> list[float]:
        if size <= 0 or len(vector) == size:
            return vector

        if len(vector) > size:
            raise ValueError(
                f"vector of dimension {len(vector)} exceeds the vector_size {size} "
                f"configured on index {self.index_qid}"
            )

        if not vector:
            raise ValueError("empty vector, nothing to pad to the index's dimension")

        return vector + [0.0] * (size - len(vector))

    def create_track(self,
        name: str,
        label: str,
        q: Content,
        additional_info: dict | None = None,
    ) -> None:
        """Stub: the vectorstore has no first class track type, each vector carries its own."""
        return None

    def get_track(self,
        name: str,
        q: Content
    ) -> Track | None:
        """Stub: tracks always "exist", they are just a per-vector label."""
        return Track(qid=q.qid, name=name, label=name)

    def create_batch(self,
        model: str,
        author: str,
        q: Content
    ) -> Batch:
        response = self.session.post(
            self._index_url("/batches"),
            json={"model": model, "author": author, "qid": q.qid},
            headers=self._get_headers(q),
            timeout=self.timeout
        )

        if not response.ok:
            self._log_response_and_raise(response)

        result = response.json()

        return Batch(
            id=str(result["batch_id"]),
            qid=q.qid,
            model=model,
            timestamp=time.time(),
            author=author,
            additional_info=result.get("additional_info", {})
        )

    def update_batch(self,
        batch_id: str,
        additional_info: dict,
        q: Content,
    ) -> None:
        response = self.session.patch(
            self._index_url(f"/batches/{batch_id}"),
            json={"additional_info": additional_info},
            headers=self._get_headers(q),
            timeout=self.timeout
        )

        if not response.ok:
            self._log_response_and_raise(response)

    def upload_tags(self, tags: list[Tag], batch_id: str, track: str, q: Content) -> None:
        if not tags:
            return

        if any(not is_vector(tag.data) for tag in tags):
            raise ValueError("a vectorstore only stores vectors, write text tags to a tagstore instead")

        # the service accepts an unbatched vector, but nothing can ever delete one
        # require that we pass a batch
        if not batch_id:
            raise ValueError("vectors must be written under a batch")

        vector_size = self._get_vector_size(q)
        num_padded = 0

        # a vector has no frame_info field, only the frame index survives the write
        vectors = []
        for tag in tags:
            data = self._fit_vector(cast(list[float], tag.data), vector_size)
            if len(data) != len(tag.data):
                num_padded += 1
            vector = {
                "track": track,
                "source": tag.source,
                "start_time": tag.start_time,
                "end_time": tag.end_time,
                "vector": data,
            }
            if tag.additional_info is not None:
                vector["additional_info"] = tag.additional_info
            if tag.frame_info is not None and "frame_idx" in tag.frame_info:
                vector["frame_idx"] = tag.frame_info["frame_idx"]
            vectors.append(vector)

        if num_padded:
            logger.warning("padded vectors to the index dimension", num_vectors=num_padded,
                           vector_size=vector_size, index_qid=self.index_qid, track=track)

        # one request is one batch over one content, so both are given once, at the top
        # level - a vector carrying its own qid or batch_id is silently dropped
        response = self.session.post(
            self._index_url("/vectors"),
            json={"qid": q.qid, "batch_id": batch_id, "vectors": vectors},
            headers=self._get_headers(q),
            timeout=self.timeout
        )

        if not response.ok:
            self._log_response_and_raise(response)

    def delete_tags_by_source(self, sources: list[str], model: str, q: Content) -> None:
        if not sources or not model:
            return

        response = self.session.delete(
            self._index_url("/vectors"),
            json={"sources": sources, "model": model},
            headers=self._get_headers(q),
            timeout=self.timeout
        )

        if not response.ok:
            self._log_response_and_raise(response)

    def delete_batch(self, batch_id: str, q: Content) -> None:
        """Delete a batch and every vector written under it."""
        response = self.session.delete(
            self._index_url(f"/batches/{batch_id}"),
            headers=self._get_headers(q),
            timeout=self.timeout
        )

        if not response.ok:
            self._log_response_and_raise(response)

    # The vectorstore is write-only from here down: a batch can be written, amended and
    # deleted but never read back, and vectors are only reachable by search. Nothing
    # reads a run back out of the index - the tagstore is the system of record for that.

    def find_tags(self, q: Content, **filters) -> list[Tag]:
        raise NotImplementedError("the vectorstore has no filtered vector listing, use search instead")

    def count_tags(self, q: Content, **filters) -> int:
        raise NotImplementedError("the vectorstore has no filtered vector listing, use search instead")

    def get_batch(self, batch_id: str, q: Content) -> Batch | None:
        raise NotImplementedError("the vectorstore does not serve vector batches for reading")

    def find_batches(self, q: Content, **filters) -> list[Batch]:
        raise NotImplementedError("the vectorstore does not serve vector batches for reading")
