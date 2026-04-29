
from collections import defaultdict
from datetime import datetime, timezone

from src.common.content import Content
from src.common.content import Content
from src.common.errors import BadRequestError, MissingResourceError
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Batch
from src.tags.track_resolver import TrackResolver
from src.status.format import *

class TaggingStatusService:
    def __init__(self,
        tagstore: Tagstore,
        track_resolver: TrackResolver,
    ):
        self.tagstore = tagstore
        self.track_resolver = track_resolver

    def get_content_summary(self, q: Content) -> ContentStatusResponse:
        batch_ids = self.tagstore.find_batches(q=q, qid=q.qid, author="tagger")

        # Collect all batches and group by track
        batches_by_track: dict[str, list[Batch]] = defaultdict(list)
        for batch_id in batch_ids:
            batch = self.tagstore.get_batch(batch_id, q=q)
            if batch is None or "tagger" not in batch.additional_info:
                continue
            batches_by_track[batch.track].append(batch)

        # Build a summary per track, collating across all batches
        model_summaries: list[ModelStatus] = []
        for track_name, batches in batches_by_track.items():
            latest_batch = max(batches, key=lambda b: b.timestamp)

            model_name = self.track_resolver.reverse_resolve(track_name)

            all_sources: set[str] = set()
            tagged_sources: set[str] = set()
            for batch in batches:
                tagger_info = batch.additional_info["tagger"]
                upload_status = tagger_info.get("upload_status")
                if upload_status:
                    all_sources.update(upload_status["all_sources"])
                    tagged_sources.update(upload_status["tagged_sources"])

            if all_sources:
                percent_completion = len(tagged_sources) / len(all_sources)
            else:
                percent_completion = 0.0

            last_run_str = datetime.fromtimestamp(latest_batch.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            model_summaries.append(
                ModelStatus(
                    model=model_name,
                    track=track_name,
                    last_run=last_run_str,
                    percent_completion=percent_completion,
                )
            )

        return ContentStatusResponse(models=model_summaries)
    
    def get_model_status(
        self,
        q: Content,
        model: str,
    ) -> ModelStatusResponse:
        track_args = self.track_resolver.resolve(model)
        track_name = track_args.name

        batch_ids = self.tagstore.find_batches(q=q, qid=q.qid, author="tagger")

        batches: list[Batch] = []
        for batch_id in batch_ids:
            batch = self.tagstore.get_batch(batch_id, q=q)
            if batch is not None and batch.track == track_name and "tagger" in batch.additional_info:
                batches.append(batch)

        if not batches:
            raise MissingResourceError(f"No jobs found for model '{model}' on content '{q.qid}'")

        # --- Build per-job details ---
        jobs: list[JobDetail] = []
        max_all_sources: int = 0
        all_tagged_sources: set[str] = set()

        for batch in batches:
            time_ran = datetime.fromtimestamp(batch.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            try:
                tagger_info: dict = batch.additional_info["tagger"]

                # Upload status (optional — absent for non-video content)
                raw_upload = tagger_info.get("upload_status")
                upload_summary: JobUploadStatusSummary | None = None
                if raw_upload:
                    num_all = len(raw_upload["all_sources"])
                    if num_all > max_all_sources:
                        max_all_sources = num_all
                    num_downloaded = len(raw_upload["downloaded_sources"])
                    num_tagged = len(raw_upload["tagged_sources"])
                    all_tagged_sources.update(raw_upload["tagged_sources"])
                    upload_summary = JobUploadStatusSummary(
                        num_job_parts=num_downloaded,
                        num_tagged_parts=num_tagged,
                    )

                params = TagArgs(**tagger_info["params"])
                job_status = JobRunStatus(**tagger_info["job_status"])

                jobs.append(
                    JobDetail(
                        time_ran=time_ran,
                        source_qid=tagger_info["source_qid"],
                        params=params,
                        job_status=job_status,
                        upload_status=upload_summary,
                    )
                )
            except Exception as e:
                raise BadRequestError(f"Malformed tagger info in batch '{batch.id}': {e}") from e

        # --- Build top-level summary ---
        latest_batch = max(batches, key=lambda b: b.timestamp)
        last_run = datetime.fromtimestamp(latest_batch.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if max_all_sources > 0:
            tagging_progress = len(all_tagged_sources) / max_all_sources
        else:
            tagging_progress = 0.0

        summary = ModelStatusSummary(
            model=model,
            track=track_name,
            last_run=last_run,
            tagging_progress=tagging_progress,
            num_content_parts=max_all_sources,
        )

        return ModelStatusResponse(summary=summary, jobs=jobs)