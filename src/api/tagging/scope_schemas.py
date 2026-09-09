"""The scope argument is polymorphic, here we list the supported schemas
"""

from marshmallow import Schema, fields, validate


_DISCRIMINATOR = "Scope identifier to remove ambiguity"


class VideoScopeSchema(Schema):
    type = fields.Constant("video", dump_only=True, metadata={"description": _DISCRIMINATOR})
    stream = fields.Str(
        metadata={
            "description": (
                "Audio or video stream to tag. If this value is not set, it's value will "
                "be automatically set depending on the model and the content."
            ),
            "example": "audio_1",
        }
    )
    # in seconds
    start_time = fields.Int(
        metadata={
            "description": "Start tagging at this point in the media stream (in seconds)"
        }
    )
    end_time = fields.Int(
        metadata={
            "description": "End tagging at this point in the media stream (in seconds)",
            "example": 600,
        }
    )


class TimeRangeScopeSchema(Schema):
    """`type: "processor"`"""
    type = fields.Constant("processor", dump_only=True, metadata={"description": _DISCRIMINATOR})
    start_time = fields.Int(
        allow_none=True,
        metadata={
            "description": "Start tagging at this point in the media stream (in seconds)"
        },
    )
    end_time = fields.Int(
        allow_none=True,
        metadata={
            "description": "End tagging at this point in the media stream (in seconds)",
            "example": 600,
        },
    )
    chunk_size = fields.Int(
        metadata={"description": "Processor will be fed intervals of this size to tag"}
    )
    stream = fields.Str(metadata={"description": "Media stream"})


class AssetScopeSchema(Schema):
    type = fields.Constant("assets", dump_only=True, metadata={"description": _DISCRIMINATOR})
    assets = fields.List(
        fields.Str(),
        allow_none=True,
        metadata={
            "description": "List of asset paths to tag",
            "example": ["assets/hello1.jpg", "assets/hello2.jpg"],
        },
    )


class LiveScopeSchema(Schema):
    type = fields.Constant("livestream", dump_only=True, metadata={"description": _DISCRIMINATOR})
    stream = fields.Str(metadata={"description": "Media stream"})
    segment_length = fields.Int(
        metadata={"description": "Tagging interval size (in seconds)"}
    )
    max_duration = fields.Int(
        allow_none=True,
        metadata={
            "description": "Maximum amount of content from livestream to tag (in seconds)"
        },
    )


class TagAlignedScopeSchema(Schema):
    type = fields.Constant("tag-aligned", dump_only=True, metadata={"description": _DISCRIMINATOR})
    stream = fields.Str(metadata={"description": "Media stream"})
    start_time = fields.Int(
        metadata={
            "description": "Start tagging at this point in the media stream (in seconds)"
        }
    )
    end_time = fields.Int(
        metadata={
            "description": "End tagging at this point in the media stream (in seconds)",
            "example": 600,
        }
    )
    track = fields.Str(
        metadata={"description": "Tag track whose intervals tagging is aligned to. Alternatively, we can set track to \"1s\", \"5s\", \"Ns\" to align tagging to fixed intervals of N seconds."}
    )


# Maps the `type` discriminator to its schema. Registered in the OpenAPI spec in server.py.
SCOPE_SCHEMAS = {
    "video": VideoScopeSchema,
    "processor": TimeRangeScopeSchema,
    "assets": AssetScopeSchema,
    "livestream": LiveScopeSchema,
    "tag-aligned": TagAlignedScopeSchema,
}


def scope_component_name(scope_type: str) -> str:
    """OpenAPI component name for a scope `type`, e.g. "tag-aligned" -> "TagAlignedScope"."""
    return "".join(part.capitalize() for part in scope_type.split("-")) + "Scope"


def scope_oneof_metadata() -> dict:
    """marshmallow field metadata that renders the scope as a discriminated `oneOf`.

    apispec merges these keys into the field's OpenAPI schema, so the `scope` request
    field points explicitly at the per-type scope components registered in the spec
    (rather than a generic object). The field stays a raw dict at runtime — see
    TaggerOptionsSchema.scope.
    """
    return {
        "description": (
            "Polymorphic tagging scope, discriminated by `type`. May be partial: omitted "
            "fields are filled in server-side based on the content (live vs. static, default "
            "audio stream)."
        ),
        "oneOf": [
            {"$ref": f"#/components/schemas/{scope_component_name(t)}"}
            for t in SCOPE_SCHEMAS
        ],
    }


class ScopeSchema(Schema):
    """Documentation-only envelope: a scope is discriminated by its `type` field.

    The concrete shape is one of the per-type schemas in SCOPE_SCHEMAS.
    """
    type = fields.Str(
        validate=validate.OneOf(list(SCOPE_SCHEMAS.keys())),
        metadata={"description": "Discriminator selecting the scope variant."},
    )
