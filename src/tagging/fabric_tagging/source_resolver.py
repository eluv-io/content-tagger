from dacite import from_dict

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

    def resolve(self, q: Content, model: str) -> list[str]:
        tracks = self.track_resolver.resolve(model)

        batch_ids = self.tagstore.find_batches(q=q, qid=q.qid, author="tagger")

        batch_by_id = {bid: self.tagstore.get_batch(bid, q=q) for bid in batch_ids}

        uploaded_sources = set()
        for t in tracks:
            track_name = t.name

            for _, batch in batch_by_id.items():
                if not batch:
                    # tiny chance it could have been deleted
                    continue
                if batch is not None and batch.track == track_name and "tagger" in batch.additional_info:
                    tagger_info: dict = batch.additional_info["tagger"]
                    uploaded_sources.update(tagger_info.get("upload_status", {}).get("uploaded_sources", []))

        return sorted(uploaded_sources)