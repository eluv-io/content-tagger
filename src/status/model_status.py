from datetime import datetime, timezone

from src.api.model_status.response_format import (
    JobDetail,
    JobUploadStatusSummary,
    ModelStatusResponse,
    ModelStatusSummary,
)
from src.tagging.fabric_tagging.model import TagArgs, JobRunStatus
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Batch
from src.tags.track_resolver import TrackResolver
from src.common.content import Content
from src.common.errors import BadRequestError, MissingResourceError


def get_model_status(
    q: Content,
    model: str,
    tagstore: Tagstore,
    track_resolver: TrackResolver,
) -> ModelStatusResponse:
    track_args = track_resolver.resolve(model)
    track_name = track_args.name

    batch_ids = tagstore.find_batches(q=q, qhit=q.qhit, author="tagger")

    batches: list[Batch] = []
    for batch_id in batch_ids:
        batch = tagstore.get_batch(batch_id, q=q)
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
