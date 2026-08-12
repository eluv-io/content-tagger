import time
import uuid
from copy import deepcopy
from dataclasses import dataclass

from src.common.content import Content
from src.tags.datastore.abstract import Datastore
from src.tags.datastore.model import *

@dataclass
class _StoredVector:
    tag: Tag
    track: str

class MockVectorstore(Datastore):
    """In-memory vectorstore for a single index, used for local runs and tests.

    Mirrors the semantics of the filesystem tagstore so the two are interchangeable
    behind the `Datastore` protocol. It does serve the reads the real service refuses,
    which is what tests inspect it through.
    """

    def __init__(self, index_qid: str):
        self.index_qid = index_qid
        self.batches: dict[str, Batch] = {}
        self.vectors: dict[str, list[_StoredVector]] = {}

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
        batch = Batch(
            id=str(uuid.uuid4()),
            qid=q.qid,
            model=model,
            timestamp=time.time(),
            author=author,
            additional_info={},
        )
        self.batches[batch.id] = batch
        self.vectors[batch.id] = []
        return batch

    def update_batch(self,
        batch_id: str,
        additional_info: dict,
        q: Content,
    ) -> None:
        if batch_id not in self.batches:
            raise ValueError(f"Batch {batch_id} not found.")
        self.batches[batch_id].additional_info = deepcopy(additional_info)

    def upload_tags(self, tags: list[Tag], batch_id: str, track: str, q: Content) -> None:
        if not tags:
            return

        if any(not is_vector(tag.data) for tag in tags):
            raise ValueError("a vectorstore only stores vectors, write text tags to a tagstore instead")

        if batch_id not in self.batches:
            raise ValueError(f"Batch {batch_id} not found. Call create_batch() first.")

        for tag in tags:
            tag.id = str(uuid.uuid4())
            self.vectors[batch_id].append(_StoredVector(tag=deepcopy(tag), track=track))

    def delete_tags_by_source(self, sources: list[str], model: str, q: Content) -> None:
        if not sources or not model:
            return

        for batch in self.find_batches(q, model=model):
            self.vectors[batch.id] = [
                v for v in self.vectors[batch.id] if v.tag.source not in sources
            ]

    def find_tags(self, q: Content, **filters) -> list[Tag]:
        if 'batch_id' in filters:
            batch_ids = [filters['batch_id']]
        else:
            batch_filters = {k: filters[k] for k in ('model', 'author') if k in filters}
            batch_ids = [b.id for b in self.find_batches(q, **batch_filters)]

        track = filters.get('track')

        results: list[Tag] = []
        for batch_id in batch_ids:
            for stored in self.vectors.get(batch_id, []):
                if track is not None and stored.track != track:
                    continue
                tag = stored.tag
                if 'sources' in filters and tag.source not in filters['sources']:
                    continue
                if 'start_time_gte' in filters and tag.start_time < filters['start_time_gte']:
                    continue
                if 'start_time_lte' in filters and tag.start_time > filters['start_time_lte']:
                    continue
                results.append(deepcopy(tag))

        if 'offset' in filters:
            results = results[filters['offset']:]
        if 'limit' in filters:
            results = results[:filters['limit']]

        return results

    def find_batches(self, q: Content, **filters) -> list[Batch]:
        batches = []
        for batch in self.batches.values():
            if batch.qid != q.qid:
                continue
            if 'model' in filters and batch.model != filters['model']:
                continue
            if 'author' in filters and batch.author != filters['author']:
                continue
            if 'timestamp_gte' in filters and batch.timestamp < filters['timestamp_gte']:
                continue
            if 'timestamp_lte' in filters and batch.timestamp > filters['timestamp_lte']:
                continue
            batches.append(deepcopy(batch))

        if 'offset' in filters:
            batches = batches[filters['offset']:]
        if 'limit' in filters:
            batches = batches[:filters['limit']]

        return batches

    def get_batch(self, batch_id: str, q: Content) -> Batch | None:
        batch = self.batches.get(batch_id)
        if batch is None:
            return None
        assert batch.qid == q.qid
        return deepcopy(batch)

    def count_tags(self, q: Content, **filters) -> int:
        return len(self.find_tags(q, **filters))

    def count_batches(self, q: Content, **filters) -> int:
        return len(self.find_batches(q, **filters))

    def delete_batch(self, batch_id: str, q: Content) -> None:
        self.batches.pop(batch_id, None)
        self.vectors.pop(batch_id, None)
