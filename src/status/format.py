
from dataclasses import asdict, dataclass

from marshmallow import Schema, fields

from src.tagging.fabric_tagging.model import JobRunStatus, TagArgs

"""structs for get status by content"""

@dataclass(frozen=True)
class ModelStatus:
    model: str
    track: str
    last_run: str
    percent_completion: float


@dataclass(frozen=True)
class ContentStatusResponse:
    models: list[ModelStatus]

"""structs for get status by content + model"""

@dataclass(frozen=True)
class JobUploadStatusSummary:
    num_job_parts: int
    num_tagged_parts: int


@dataclass(frozen=True)
class JobDetail:
    time_ran: str
    source_qid: str
    params: TagArgs
    job_status: JobRunStatus
    upload_status: JobUploadStatusSummary | None


@dataclass(frozen=True)
class ModelStatusSummary:
    model: str
    track: str
    last_run: str
    tagging_progress: float
    num_content_parts: int


@dataclass(frozen=True)
class ModelStatusResponse:
    summary: ModelStatusSummary
    jobs: list[JobDetail]


# --- marshmallow schemas ---

class ModelStatusSchema(Schema):
    """High-level tagging status for one model on the content."""
    model = fields.Str(metadata={"description": "Model identifier", "example": "celeb"})
    track = fields.Str(
        metadata={
            "description": "Track the model writes its tags to",
            "example": "celebrity_detection",
        }
    )
    last_run = fields.Str(
        metadata={
            "description": "ISO 8601 timestamp of the most recent job run",
            "example": "2026-02-25T18:30:00Z",
        }
    )
    percent_completion = fields.Float(
        metadata={
            "description": "Ratio of content parts that have been tagged (0.0 – 1.0)",
            "example": 0.85,
        }
    )


class ContentStatusResponseSchema(Schema):
    """Summary of tagging status across all models for a content object."""
    models = fields.List(
        fields.Nested(ModelStatusSchema),
        metadata={"description": "List of per-model status summaries"},
    )


class JobUploadStatusSummarySchema(Schema):
    """Summary of how many parts were uploaded for a job."""
    num_job_parts = fields.Int(
        metadata={"description": "Total number of parts in the job", "example": 60}
    )
    num_tagged_parts = fields.Int(
        metadata={
            "description": "Number of parts that were successfully tagged and uploaded",
            "example": 60,
        }
    )


class JobRunStatusSchema(Schema):
    """Status of a completed job run."""
    status = fields.Str(
        metadata={"description": "Final status of the job", "example": "Completed"}
    )
    time_ran = fields.Str(
        metadata={
            "description": "Duration the job ran before stopping",
            "example": "1h 0m 0s",
        }
    )


class JobDetailSchema(Schema):
    """Details of a single tagging job run."""
    time_ran = fields.Str(
        metadata={
            "description": "Duration that the job ran before stopping",
            "example": "1h 0m 0s",
        }
    )
    source_qid = fields.Str(
        metadata={
            "description": "Content object id that was tagged",
            "example": "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm",
        }
    )
    # params is a TagArgs dataclass with a nested polymorphic Scope; serialize it
    # recursively with asdict to preserve the existing JSON shape exactly.
    params = fields.Function(
        serialize=lambda obj: asdict(obj.params),
        metadata={"description": "Parameters that were used to run the tagging job."},
    )
    job_status = fields.Nested(JobRunStatusSchema)
    upload_status = fields.Nested(
        JobUploadStatusSummarySchema,
        allow_none=True,
        metadata={
            "description": (
                "Upload progress for this job. Null if the job did not reach the upload "
                "phase."
            )
        },
    )


class ModelStatusSummarySchema(Schema):
    """Aggregate summary for a model's tagging progress on a content object."""
    model = fields.Str(metadata={"description": "Model identifier", "example": "celeb"})
    track = fields.Str(
        metadata={
            "description": "Track the model writes its tags to",
            "example": "celebrity_detection",
        }
    )
    last_run = fields.Str(
        metadata={
            "description": "ISO 8601 timestamp of the most recent job run",
            "example": "2026-02-25T18:30:00Z",
        }
    )
    tagging_progress = fields.Float(
        metadata={
            "description": "Fraction of content parts that have been tagged (0.0 – 1.0)",
            "example": 0.85,
        }
    )
    num_content_parts = fields.Int(
        metadata={"description": "Total number of parts in the content", "example": 120}
    )


class ModelStatusResponseSchema(Schema):
    """Detailed status for a single model on a content object."""
    summary = fields.Nested(ModelStatusSummarySchema)
    jobs = fields.List(
        fields.Nested(JobDetailSchema),
        metadata={"description": "Chronological list of every job that has been run for this model"},
    )
