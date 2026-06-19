
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
    destination_qid = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={
            "description": (
                "Content where tags should be written to. Leaving empty will write to the same content the model is running on."
            )
        },
    )
    replace = fields.Bool(
        load_default=None,
        allow_none=True,
        metadata={
            "description": (
                "Replace already existing tagstore tags. Set to true to retag, set to "
                "false to enable diff-based tagging."
            )
        },
    )
    max_fetch_retries = fields.Int(
        load_default=None,
        allow_none=True,
        metadata={
            "description": (
                "If a fetch operation fails, we can retry this many times before the job "
                "is marked as failed."
            )
        },
    )
    # Dict (validates the value is an object) + oneOf metadata so the OpenAPI spec points
    # explicitly at the per-type scope schemas. Stays a plain dict at runtime so
    # ArgsResolver's merge/resolution pipeline is unchanged; it may be partial, so the
    # individual scope fields are not required here.
    scope = fields.Dict(load_default=dict, metadata=scope_oneof_metadata())

    @post_load
    def make(self, data, **kwargs):
        return TaggerOptions(**data)


class JobSpecSchema(Schema):
    model = fields.Str(
        required=True,
        metadata={"description": "Name of model to run", "example": "asr"},
    )
    model_params = fields.Dict(
        load_default=dict,
        metadata={"description": "Unstructured model level parameters", "example": {"fps": 1.5, "min_confidence": 0.8}},
    )
    track_suffix = fields.Str(
        load_default="",
        metadata={
            "description": (
                "If set this will be appended to the track name that the tagger "
                "writes to. This allows you to write to different tracks with the same "
                "model by specifying different suffixes."
            ),
            "example": "test run"
        },
    )
    caller_info = fields.Dict(
        keys=fields.Str(),
        values=fields.Str(),
        load_default=dict,
        metadata={
            "description": (
                "Arbitrary key-value pairs that will be stored with the job for later "
                "reference."
            ),
            "example": {"user_note": "This is a note about the job"},
        },
    )
    overrides = fields.Nested(
        TaggerOptionsSchema,
        load_default=TaggerOptions,
        metadata={
            "description": "Per-job options that override the request-level options.",
            "example": {
                "scope": {
                    "stream": "spanish_audio_5_1"
                }
            }
        },
    )

    @post_load
    def make(self, data, **kwargs):
        return JobSpec(**data)


class StartJobsRequestSchema(Schema):
    options = fields.Nested(
        TaggerOptionsSchema,
        load_default=TaggerOptions,
        metadata={"description": "Knobs that apply to all tagger jobs in the request."},
    )
    jobs = fields.List(
        fields.Nested(JobSpecSchema),
        load_default=list,
        metadata={"description": "List of individual job parameters."},
    )

    @post_load
    def make(self, data, **kwargs):
        return StartJobsRequest(**data)


class StatusRequestSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    start = fields.Int(load_default=0, metadata={"description": "Pagination offset."})
    limit = fields.Int(
        load_default=None,
        allow_none=True,
        metadata={"description": "Maximum number of jobs to return. Null for no limit."},
    )
    qid = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={"description": "Filter by content id."},
    )
    status = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={"description": "Filter by job status."},
    )
    tenant = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={"description": "Filter by tenant. Requires tenant admin privileges."},
    )
    user = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={
            "description": (
                "Filter by user address. Provided authorization must match the authenticated user unless "
                "querying as tenant admin."
            )
        },
    )
    model = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={"description": 'Filter by model name (e.g. "asr", "celeb").'},
    )
    title = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={
            "description": "Filter by content title (case-insensitive substring match)."
        },
    )

    @post_load
    def make(self, data, **kwargs):
        return StatusRequest(**data)