
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
    model = fields.Str()
    track = fields.Str()
    last_run = fields.Str()
    percent_completion = fields.Float()


class ContentStatusResponseSchema(Schema):
    models = fields.List(fields.Nested(ModelStatusSchema))


class JobUploadStatusSummarySchema(Schema):
    num_job_parts = fields.Int()
    num_tagged_parts = fields.Int()


class JobRunStatusSchema(Schema):
    status = fields.Str()
    time_ran = fields.Str()


class JobDetailSchema(Schema):
    time_ran = fields.Str()
    source_qid = fields.Str()
    # params is a TagArgs dataclass with a nested polymorphic Scope; serialize it
    # recursively with asdict to preserve the existing JSON shape exactly.
    params = fields.Function(serialize=lambda obj: asdict(obj.params))
    job_status = fields.Nested(JobRunStatusSchema)
    upload_status = fields.Nested(JobUploadStatusSummarySchema, allow_none=True)


class ModelStatusSummarySchema(Schema):
    model = fields.Str()
    track = fields.Str()
    last_run = fields.Str()
    tagging_progress = fields.Float()
    num_content_parts = fields.Int()


class ModelStatusResponseSchema(Schema):
    summary = fields.Nested(ModelStatusSummarySchema)
    jobs = fields.List(fields.Nested(JobDetailSchema))
