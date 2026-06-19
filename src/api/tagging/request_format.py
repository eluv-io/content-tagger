
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypeAlias

from marshmallow import EXCLUDE, Schema, fields, post_load

from src.api.tagging.scope_schemas import scope_oneof_metadata

@dataclass
class TaggerOptions:
    destination_qid: str | None = None
    replace: bool | None = None
    max_fetch_retries: int | None = None
    # unstructured dict to allow for flexible scope definitions - will be parsed into a ScopeDTO
    scope: dict[str, Any] = field(default_factory=dict)

@dataclass
class JobSpec:
    model: str
    model_params: dict[str, Any] = field(default_factory=dict)
    track_suffix: str = ""
    caller_info: dict[str, str] = field(default_factory=dict)
    overrides: TaggerOptions = field(default_factory=TaggerOptions)

@dataclass
class StartJobsRequest:
    options: TaggerOptions = field(default_factory=TaggerOptions)
    jobs: list[JobSpec] = field(default_factory=list)

@dataclass
class StatusRequest:
    start: int = 0
    limit: int | None = None
    qid: str | None = None
    status: str | None = None
    tenant: str | None = None
    user: str | None = None
    model: str | None = None
    title: str | None = None


# --- marshmallow schemas ---
# For validating request and generating open api docs, these get converted into raw dataclasses listed above

class TaggerOptionsSchema(Schema):
    destination_qid = fields.Str(load_default=None, allow_none=True)
    replace = fields.Bool(load_default=None, allow_none=True)
    max_fetch_retries = fields.Int(load_default=None, allow_none=True)
    # Dict (validates the value is an object) + oneOf metadata so the OpenAPI spec points
    # explicitly at the per-type scope schemas. Stays a plain dict at runtime so
    # ArgsResolver's merge/resolution pipeline is unchanged; it may be partial, so the
    # individual scope fields are not required here.
    scope = fields.Dict(load_default=dict, metadata=scope_oneof_metadata())

    @post_load
    def make(self, data, **kwargs):
        return TaggerOptions(**data)


class JobSpecSchema(Schema):
    model = fields.Str(required=True)
    model_params = fields.Dict(load_default=dict)
    track_suffix = fields.Str(load_default="")
    caller_info = fields.Dict(keys=fields.Str(), values=fields.Str(), load_default=dict)
    overrides = fields.Nested(TaggerOptionsSchema, load_default=TaggerOptions)

    @post_load
    def make(self, data, **kwargs):
        return JobSpec(**data)


class StartJobsRequestSchema(Schema):
    options = fields.Nested(TaggerOptionsSchema, load_default=TaggerOptions)
    jobs = fields.List(fields.Nested(JobSpecSchema), load_default=list)

    @post_load
    def make(self, data, **kwargs):
        return StartJobsRequest(**data)


class StatusRequestSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    start = fields.Int(load_default=0)
    limit = fields.Int(load_default=None, allow_none=True)
    qid = fields.Str(load_default=None, allow_none=True)
    status = fields.Str(load_default=None, allow_none=True)
    tenant = fields.Str(load_default=None, allow_none=True)
    user = fields.Str(load_default=None, allow_none=True)
    model = fields.Str(load_default=None, allow_none=True)
    title = fields.Str(load_default=None, allow_none=True)

    @post_load
    def make(self, data, **kwargs):
        return StatusRequest(**data)