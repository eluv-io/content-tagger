
from src.tags.datastore.abstract import Datastore
from src.tags.vectorstore.model import VectorstoreConfig
from src.tags.vectorstore.mock_vectorstore import MockVectorstore
from src.tags.vectorstore.rest_vectorstore import RestVectorstore

class VectorstoreFactory:
    """Builds a `Datastore` bound to a vector index.

    Unlike a tagstore, a vectorstore is addressed per index, so the store can only be
    built once the caller's index_qid is known.
    """

    def __init__(self, cfg: VectorstoreConfig):
        self.cfg = cfg
        # mocks are kept per index so their contents survive across jobs
        self._mocks: dict[str, MockVectorstore] = {}

    def create(self, index_qid: str) -> Datastore:
        if self.cfg.base_url:
            return RestVectorstore(self.cfg.base_url, self.cfg.timeout, index_qid)
        return self._mocks.setdefault(index_qid, MockVectorstore(index_qid))
