"""The scope argument is polymorphic, here we list the supported schemas
"""

from marshmallow import Schema, fields, validate


class VideoScopeSchema(Schema):
    type = fields.Constant("video", dump_only=True)
    stream = fields.Str()
    # in seconds
    start_time = fields.Int()
    end_time = fields.Int()


class TimeRangeScopeSchema(Schema):
    """`type: "processor"`"""
    type = fields.Constant("processor", dump_only=True)
    start_time = fields.Int(allow_none=True)
    end_time = fields.Int(allow_none=True)
    chunk_size = fields.Int()
    stream = fields.Str()


class AssetScopeSchema(Schema):
    type = fields.Constant("assets", dump_only=True)
    assets = fields.List(fields.Str(), allow_none=True)


class LiveScopeSchema(Schema):
    type = fields.Constant("livestream", dump_only=True)
    stream = fields.Str()
    segment_length = fields.Int()
    max_duration = fields.Int(allow_none=True)


class TagAlignedScopeSchema(Schema):
    type = fields.Constant("tag-aligned", dump_only=True)
    stream = fields.Str()
    start_time = fields.Int()
    end_time = fields.Int()
    track = fields.Str()


# Maps the `type` discriminator to its schema. Registered in the OpenAPI spec in server.py.
SCOPE_SCHEMAS = {
    "video": VideoScopeSchema,
    "processor": TimeRangeScopeSchema,
    "assets": AssetScopeSchema,
    "livestream": LiveScopeSchema,
    "tag-aligned": TagAlignedScopeSchema,
}


class ScopeSchema(Schema):
    """Documentation-only envelope: a scope is discriminated by its `type` field.

    The concrete shape is one of the per-type schemas in SCOPE_SCHEMAS.
    """
    type = fields.Str(
        validate=validate.OneOf(list(SCOPE_SCHEMAS.keys())),
        metadata={"description": "Discriminator selecting the scope variant."},
    )
