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

        uploaded_sources = set()
        for t in tracks:
            track_name = t.name

            batch_ids = self.tagstore.find_batches(q=q, qid=q.qid, author="tagger")

            
            for batch_id in batch_ids:
                batch = self.tagstore.get_batch(batch_id, q=q)
                if batch is not None and batch.track == track_name and "tagger" in batch.additional_info:
                    tagger_info: dict = batch.additional_info["tagger"]
                    uploaded_sources.update(tagger_info.get("upload_status", {}).get("uploaded_sources", []))

        return sorted(uploaded_sources)