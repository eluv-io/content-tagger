
from collections import defaultdict

from src.api.content_status.response_format import ContentStatusResponse, ModelStatusSummary
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Batch
from src.tags.track_resolver import TrackResolver

def get_content_summary(qid: str, tagstore: Tagstore, track_resolver: TrackResolver) -> ContentStatusResponse:
    batch_ids = tagstore.find_batches(qhit=qid, author="tagger")

    # Collect all batches and group by track
    batches_by_track: dict[str, list[Batch]] = defaultdict(list)
    for batch_id in batch_ids:
        batch = tagstore.get_batch(batch_id)
        if batch is None:
            continue
        batches_by_track[batch.track].append(batch)

    # Build a summary per track, collating across all batches
    model_summaries: list[ModelStatusSummary] = []
    for track_name, batches in batches_by_track.items():
        latest_batch = max(batches, key=lambda b: b.timestamp)

        model_name = track_resolver.reverse_resolve(track_name)

        all_sources: set[str] = set()
        tagged_sources: set[str] = set()
        for batch in batches:
            tagger_info = batch.additional_info.get("tagger", {})
            upload_status = tagger_info.get("upload_status")
            if upload_status:
                all_sources.update(upload_status.get("all_sources", []))
                tagged_sources.update(upload_status.get("tagged_sources", []))

        if all_sources:
            percent_completion = len(tagged_sources) / len(all_sources)
        else:
            percent_completion = 0.0

        model_summaries.append(
            ModelStatusSummary(
                model=model_name,
                track=track_name,
                last_run=latest_batch.timestamp,
                percent_completion=percent_completion,
            )
        )

    return ContentStatusResponse(models=model_summaries)