import json
import time
import requests
from dateutil import parser

from src.common.content import Content
from src.common.logging import logger
from src.tags.datastore.abstract import Datastore
from src.tags.datastore.model import Batch, Tag, Track, is_vector

class RestVectorstore(Datastore):
    """Datastore backed by the elv-vectorstore service, bound to a single index.

    `index_qid` is the content that holds the index; the `q` passed to each method is
    the content being embedded, which is what a vector's `qid` refers to.

    The batch read/update endpoints mirror the tagstore's (`vector-batches/{id}`) and
    are not part of the published vectorstore spec yet.
    """

    def __init__(self, base_url: str, timeout: int, index_qid: str):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.index_qid = index_qid
        self.session = requests.Session()

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
            self._index_url("/vector-batches"),
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
            self._index_url(f"/vector-batches/{batch_id}"),
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

        # the vectorstore schema has no additional_info on a vector, so only the
        # frame index survives the write
        vectors = []
        for tag in tags:
            vector = {
                "batch_id": batch_id,
                "qid": q.qid,
                "track": track,
                "source": tag.source,
                "start_time": tag.start_time,
                "end_time": tag.end_time,
                "vector": tag.data,
            }
            if tag.frame_info is not None and "frame_idx" in tag.frame_info:
                vector["frame_idx"] = tag.frame_info["frame_idx"]
            vectors.append(vector)

        response = self.session.post(
            self._index_url("/vectors"),
            json={"vectors": vectors},
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

    def find_batches(self, q: Content, **filters) -> list[Batch]:
        """
        Find vector batches with flexible filtering.

        Supported filters:
        - model: str
        - author: str
        - limit: int
        - offset: int
        """
        params = {}
        for key in ('model', 'author', 'limit'):
            if key in filters:
                params[key] = filters[key]
        if 'offset' in filters:
            params['start'] = filters['offset']

        response = self.session.get(
            self._index_url("/vector-batches"),
            params=params,
            headers=self._get_headers(q),
            timeout=self.timeout
        )

        if not response.ok:
            self._log_response_and_raise(response)

        batches = response.json().get('batches', [])

        # the index can span several contents, only this one's batches are relevant
        return [self._parse_batch(b) for b in batches if b.get("qid", q.qid) == q.qid]

    def get_batch(self, batch_id: str, q: Content) -> Batch | None:
        response = self.session.get(
            self._index_url(f"/vector-batches/{batch_id}"),
            headers=self._get_headers(q),
            timeout=self.timeout
        )

        if response.status_code == 404:
            return None

        if not response.ok:
            self._log_response_and_raise(response)

        return self._parse_batch(response.json())

    def find_tags(self, q: Content, **filters) -> list[Tag]:
        raise NotImplementedError("the vectorstore has no filtered vector listing, use search instead")

    def count_tags(self, q: Content, **filters) -> int:
        raise NotImplementedError("the vectorstore has no filtered vector listing, use search instead")

    def delete_batch(self, batch_id: str, q: Content) -> None:
        raise NotImplementedError("the vectorstore deletes vectors by model and source, not by batch")

    def _parse_batch(self, batch_data: dict) -> Batch:
        return Batch(
            id=str(batch_data['id']),
            qid=batch_data['qid'],
            model=batch_data['model'],
            timestamp=parser.isoparse(batch_data['created_at'].replace("Z", "+00:00")).timestamp(),
            author=batch_data['author'],
            additional_info=batch_data.get("additional_info", {})
        )
