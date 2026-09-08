from dataclasses import dataclass

from marshmallow import Schema, fields

from src.service.model import TagDetails

@dataclass(frozen=True)
class StartStatus:
    job_id: str
    model: str
    started: bool
    message: str
    dependencies: list[str]
    error: str | None

@dataclass(frozen=True)
class StartTaggingResponse:
    jobs: list[StartStatus]

@dataclass(frozen=True)
class JobStatus: 
    qid: str
    job_id: str
    status: str
    model: str
    stream: str
    created_at: str
    params: dict
    tenant: str
    user: str
    title: str
    tagging_progress: str
    # between 0 and 1
    progress: float
    tagged_duration: float
    error: str | None
    tag_details: TagDetails | None
    # whether the content being tagged is a livestream
    is_live: bool

@dataclass(frozen=True)
class StatusMeta:
    total: int
    start: int
    limit: int | None
    count: int

@dataclass(frozen=True)
class StatusResponse:
    jobs: list[JobStatus]
    meta: StatusMeta

@dataclass(frozen=True)
class StopStatus:
    job_id: str
    message: str

@dataclass(frozen=True)
class StopTaggingResponse:
    jobs: list[StopStatus]
    message: str


# --- marshmallow schemas ---
# (included for generating open api docs)
# After the handler runs, flask-smorest uses these to serialize the above dataclasses

class StartStatusSchema(Schema):
    job_id = fields.Str(metadata={"description": "Job id", "example": "1234"})
    model = fields.Str(metadata={"description": "Model name", "example": "asr"})
    started = fields.Bool(
        metadata={"description": "Did the job start yes or no", "example": True}
    )
    message = fields.Str(
        metadata={
            "description": "Additional information or words of positive affirmation",
            "example": "successfully started",
        }
    )
    dependencies = fields.List(
        fields.Str(),
        metadata={"description": "Job ids that this job depends on. The job will not start until these have completed."},
    )
    error = fields.Str(
        allow_none=True,
        metadata={"description": "Error message if the job failed", "example": None},
    )


class StartTaggingResponseSchema(Schema):
    jobs = fields.List(
        fields.Nested(StartStatusSchema),
        metadata={"description": "Start status for each job in the request"},
    )


class WarningResponseSchema(Schema):
    """Summary of warnings encountered during tagging."""
    num_warnings = fields.Int(
        metadata={"description": "Total number of warnings", "example": 25}
    )
    last_warning = fields.Str(
        metadata={
            "description": "Most recent warning message",
            "example": (
                "Failed to download part "
                "hqpe2EChvsWa6KNqFBqzn5yit7Z8NFLZ6DdxbcykW673Musy6JL4Vj"
            ),
        }
    )


class TagDetailsSchema(Schema):
    """Detailed progress information for the tagging job."""
    tag_status = fields.Str(
        metadata={
            "description": "Human-readable description of what the job is currently doing",
            "example": "Fetching content",
        }
    )
    time_running = fields.Float(
        metadata={
            "description": "How long the job has been running (in seconds)",
            "example": 70.05,
        }
    )
    progress = fields.Float(
        metadata={
            "description": "Overall job progress as a ratio from 0.0 to 1.0",
            "example": 0.56,
        }
    )
    tagging_progress = fields.Str(
        metadata={
            "description": "Ratio of tagged parts to total parts (deprecated)",
            "example": "120/214",
        }
    )
    tagged_duration = fields.Float(
        metadata={
            "description": "Number of seconds tagged",
            "example": 2000.0,
        }
    )
    total_parts = fields.Int(
        metadata={
            "description": "Total number of parts in scope for the tag request",
            "example": 214,
        }
    )
    downloaded_parts = fields.Int(
        metadata={
            "description": "Number of parts that have been downloaded",
            "example": 150,
        }
    )
    tagged_parts = fields.Int(
        metadata={"description": "Number of parts that have been tagged", "example": 120}
    )
    warnings = fields.Nested(
        WarningResponseSchema,
        allow_none=True,
        metadata={"description": "Warning summary, or null if there are no warnings"},
    )


class JobStatusSchema(Schema):
    qid = fields.Str(
        metadata={
            "description": "Content object id that is being tagged",
            "example": "iq__3Bs4xdLS9obhmQVWHNJvJ457ZL2X",
        }
    )
    job_id = fields.Str(
        metadata={
            "description": "Job id",
            "example": "efd8e379-9256-40bc-9aa6-1fab74234f85",
        }
    )
    status = fields.Str(
        metadata={"description": "Current state of the job", "example": "running"}
    )
    model = fields.Str(metadata={"description": "Name of model", "example": "asr"})
    stream = fields.Str(metadata={"description": "Stream name", "example": "video"})
    created_at = fields.Str(
        metadata={
            "description": "ISO 8601 timestamp of when the job was created",
            "example": "2026-03-25T03:33:38.746127",
        }
    )
    params = fields.Dict(
        metadata={"description": "Parameters that were used to run the tagging job."}
    )
    tenant = fields.Str(
        metadata={
            "description": "Tenant identifier",
            "example": "iten2fY3bSh8Q7zY1t6vGZ8mUNFLHAWM",
        }
    )
    user = fields.Str(
        metadata={
            "description": "Address of the user who submitted the job",
            "example": "0x9d7186b18ecbb5751719795415e9b8146e1bed2b",
        }
    )
    title = fields.Str(
        metadata={"description": "Title of the content being tagged", "example": "Casablanca"}
    )
    tagging_progress = fields.Str(
        metadata={
            "description": (
                "Deprecated: ratio of tagged_parts to total_parts. Use progress instead."
            ),
            "example": "120/214",
            "deprecated": True,
        }
    )
    progress = fields.Float(
        metadata={
            "description": "Overall job progress as a fraction (0.0 – 1.0)",
            "example": 0.56,
        }
    )
    error = fields.Str(
        allow_none=True,
        metadata={
            "description": "Error message if the job failed or was cancelled",
            "example": None,
        },
    )
    is_live = fields.Bool(
        metadata={
            "description": "Whether the content being tagged is a livestream",
            "example": False,
        }
    )
    tag_details = fields.Nested(
        TagDetailsSchema,
        allow_none=True,
        metadata={
            "description": (
                "Detailed progress information for the tagging job. Null if the job is "
                "still on the queue."
            )
        },
    )


class StatusMetaSchema(Schema):
    """Pagination metadata for list responses."""
    total = fields.Int(
        metadata={"description": "Total number of matching items", "example": 8}
    )
    start = fields.Int(
        metadata={"description": "Offset of the first item returned", "example": 0}
    )
    limit = fields.Int(
        allow_none=True,
        metadata={
            "description": "Maximum number of items requested, or null for no limit",
            "example": None,
        },
    )
    count = fields.Int(
        metadata={
            "description": "Number of items returned in this response",
            "example": 8,
        }
    )


class StatusResponseSchema(Schema):
    jobs = fields.List(
        fields.Nested(JobStatusSchema),
        metadata={"description": "Individual job statuses"},
    )
    meta = fields.Nested(StatusMetaSchema)


class StopStatusSchema(Schema):
    job_id = fields.Str(metadata={"description": "Job ID", "example": "1234"})
    message = fields.Str(metadata={"description": "Message"})


class StopTaggingResponseSchema(Schema):
    jobs = fields.List(
        fields.Nested(StopStatusSchema),
        metadata={"description": "Stop statuses of individual jobs"},
    )
    message = fields.Str(metadata={"description": "Top level message"})