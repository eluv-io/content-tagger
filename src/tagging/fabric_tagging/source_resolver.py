from src.common.content import Content
from src.tags.datastore.abstract import Datastore
from src.tags.track_resolver import TrackResolver
from src.tags.vectorstore.factory import VectorstoreFactory

class SourceResolver:
    def __init__(
        self,
        tagstore: Datastore,
        vectorstores: VectorstoreFactory,
        track_resolver: TrackResolver
    ):
        self.tagstore = tagstore
        self.vectorstores = vectorstores
        self.track_resolver = track_resolver

    def resolve(self, q: Content, model: str, track_suffix: str = "", index_qid: str = "") -> list[str]:
        """Return the sources already tagged by a previous run of this (model,
        track_suffix), so a non-replace run can skip them.

        A vector model records its report against the index rather than the tagstore,
        so the index is consulted too when one is given.
        """
        model_name = self.track_resolver.apply_suffix(model, track_suffix)

        stores = [self.tagstore]
        if index_qid:
            stores.append(self.vectorstores.create(index_qid))

        uploaded_sources = set()
        for store in stores:
            for batch in store.find_batches(q=q, qid=q.qid, author="tagger", model=model_name):
                if "tagger" in batch.additional_info:
                    tagger_info: dict = batch.additional_info["tagger"]
                    uploaded_sources.update(tagger_info.get("upload_status", {}).get("uploaded_sources", []))

        return sorted(uploaded_sources)
