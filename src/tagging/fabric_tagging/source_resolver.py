from src.common.content import Content
from src.tags.tagstore.abstract import Tagstore
from src.tags.track_resolver import TrackResolver

class SourceResolver:
    def __init__(
        self,
        tagstore: Tagstore,
        track_resolver: TrackResolver
    ):
        self.tagstore = tagstore
        self.track_resolver = track_resolver

    def resolve(self, q: Content, model: str, track_suffix: str = "") -> list[str]:
        """Return the sources already tagged by a previous run of this (model,
        track_suffix), so a non-replace run can skip them.
        """
        model_name = self.track_resolver.apply_suffix(model, track_suffix)

        batch_ids = self.tagstore.find_batches(q=q, qid=q.qid, author="tagger", model=model_name)

        uploaded_sources = set()
        for bid in batch_ids:
            batch = self.tagstore.get_batch(bid, q=q)
            if batch is None:
                # tiny chance it could have been deleted
                continue
            if "tagger" in batch.additional_info:
                tagger_info: dict = batch.additional_info["tagger"]
                uploaded_sources.update(tagger_info.get("upload_status", {}).get("uploaded_sources", []))

        return sorted(uploaded_sources)
