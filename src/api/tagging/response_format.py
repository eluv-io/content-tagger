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
    error: str | None
    tag_details: TagDetails | None

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
    job_id = fields.Str()
    model = fields.Str()
    started = fields.Bool()
    message = fields.Str()
    dependencies = fields.List(fields.Str())
    error = fields.Str(allow_none=True)


class StartTaggingResponseSchema(Schema):
    jobs = fields.List(fields.Nested(StartStatusSchema))


class WarningResponseSchema(Schema):
    num_warnings = fields.Int()
    last_warning = fields.Str()


class TagDetailsSchema(Schema):
    tag_status = fields.Str()
    time_running = fields.Float()
    progress = fields.Float()
    tagging_progress = fields.Str()
    total_parts = fields.Int()
    downloaded_parts = fields.Int()
    tagged_parts = fields.Int()
    warnings = fields.Nested(WarningResponseSchema, allow_none=True)


class JobStatusSchema(Schema):
    qid = fields.Str()
    job_id = fields.Str()
    status = fields.Str()
    model = fields.Str()
    stream = fields.Str()
    created_at = fields.Str()
    params = fields.Dict()
    tenant = fields.Str()
    user = fields.Str()
    title = fields.Str()
    tagging_progress = fields.Str()
    progress = fields.Float()
    error = fields.Str(allow_none=True)
    tag_details = fields.Nested(TagDetailsSchema, allow_none=True)


class StatusMetaSchema(Schema):
    total = fields.Int()
    start = fields.Int()
    limit = fields.Int(allow_none=True)
    count = fields.Int()


class StatusResponseSchema(Schema):
    jobs = fields.List(fields.Nested(JobStatusSchema))
    meta = fields.Nested(StatusMetaSchema)


class StopStatusSchema(Schema):
    job_id = fields.Str()
    message = fields.Str()


class StopTaggingResponseSchema(Schema):
    jobs = fields.List(fields.Nested(StopStatusSchema))
    message = fields.Str()