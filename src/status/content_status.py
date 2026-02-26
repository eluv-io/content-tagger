
from collections import defaultdict
from datetime import datetime, timezone

from src.api.content_status.response_format import ContentStatusResponse, ModelStatusSummary
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Batch
from src.tags.track_resolver import TrackResolver
from src.common.content import Content

def get_content_summary(q: Content, tagstore: Tagstore, track_resolver: TrackResolver) -> ContentStatusResponse:
    batch_ids = tagstore.find_batches(q=q, qhit=q.qhit, author="tagger")

    # Collect all batches and group by track
    batches_by_track: dict[str, list[Batch]] = defaultdict(list)
    for batch_id in batch_ids:
        batch = tagstore.get_batch(batch_id, q=q)
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

        last_run_str = datetime.fromtimestamp(latest_batch.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        model_summaries.append(
            ModelStatusSummary(
                model=model_name,
                track=track_name,
                last_run=last_run_str,
                percent_completion=percent_completion,
            )
        )

    return ContentStatusResponse(models=model_summaries)