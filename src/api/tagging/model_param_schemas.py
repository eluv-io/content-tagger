"""`model_params` is polymorphic: each model accepts its own set of parameters.

This mirrors the scope schemas (see scope_schemas.py): per-model marshmallow
schemas, a registry, and a `oneOf` metadata helper. The schemas are documentation
only -- `model_params` stays a raw dict at runtime (see JobSpecSchema.model_params),
so nothing in the resolution pipeline changes.

Unlike scope, the discriminator (`model`) is a *sibling* field of `model_params`
rather than a field inside it, so we render a plain `oneOf` with a descriptive note
rather than a formal OpenAPI `discriminator` mapping.

The fields below mirror the params dataclasses in params.py. Fields with a default
render that default in the docs; fields without one are marked required.
"""

from marshmallow import Schema, fields


class ParamsAsrSchema(Schema):
    """`model: "asr"`"""
    word_level = fields.Bool(
        load_default=True,
        metadata={
            "description": "Whether to output word level tags or collapse into longer tags.",
        },
    )
    prettify = fields.Bool(
        load_default=True,
        metadata={"description": "Punctuation/capitalization postprocessing."},
    )
    pretty_trail = fields.Bool(
        load_default=True,
        metadata={
            "description": (
                'If running at word level, output a second track "auto_captions" which '
                "processes bigger blocks of audio in a single pass for better output quality "
                "and returns sentence level tags. This can be particularly useful for live "
                'tagging so you can have a "pretty" track alongside a live-edge track.'
            ),
        },
    )
    pretty_trail_buffer = fields.Int(
        load_default=30,
        metadata={
            "description": (
                "(when pretty_trail=true) how many seconds worth of audio do we wait for "
                "before we output a batch of auto_captions tags."
            ),
        },
    )


class ParamsMultilingualAsrSchema(Schema):
    """`model: "multilingual_asr"`"""
    word_level = fields.Bool(
        load_default=True,
        metadata={
            "description": "Whether to output word level tags or collapse into longer tags.",
        },
    )
    prettify = fields.Bool(
        load_default=True,
        metadata={"description": "Punctuation/capitalization postprocessing."},
    )
    pretty_trail = fields.Bool(
        load_default=True,
        metadata={
            "description": (
                'If running at word level, output a second track "auto_captions" which '
                "processes bigger blocks of audio in a single pass for better output quality "
                "and returns sentence level tags. This can be particularly useful for live "
                'tagging so you can have a "pretty" track alongside a live-edge track.'
            ),
        },
    )
    pretty_trail_buffer = fields.Int(
        load_default=30,
        metadata={
            "description": (
                "(when pretty_trail=true) how many seconds worth of audio do we wait for "
                "before we output a batch of auto_captions tags."
            ),
        },
    )


class ParamsLlavaSchema(Schema):
    """`model: "llava"` (frame description)"""
    fps = fields.Int(
        required=True,
        metadata={"description": "Frequency to generate frame level tags.", "example": 1},
    )
    model = fields.Str(
        required=True,
        metadata={"description": "Ollama model to run.", "example": "llava"},
    )
    temperature = fields.Float(
        required=True,
        metadata={"description": "Sampling temperature for generation.", "example": 0.2},
    )
    prompt = fields.Str(
        required=True,
        metadata={
            "description": "Prompt used to generate the frame description.",
            "example": "Describe this frame.",
        },
    )


class ParamsCelebSchema(Schema):
    """`model: "celeb"`"""
    fps = fields.Float(
        load_default=4,
        metadata={"description": "Frames per second to sample for face detection."},
    )
    thres = fields.Float(
        load_default=0.4,
        metadata={"description": "Minimum confidence threshold."},
    )
    min_box_size = fields.Float(
        load_default=0,
        metadata={
            "description": (
                "Minimum face bounding box size to consider. Value is a ratio of the area "
                "of the full frame."
            ),
        },
    )
    allow_single_frame = fields.Bool(
        load_default=False,
        metadata={
            "description": (
                "Whether to generate a video tag if we only detect a celebrity in a single "
                "frame. Setting to true helps to avoid false positives for erroneous "
                "single-frame detections."
            ),
        },
    )
    ground_truth = fields.Str(
        metadata={
            "description": "Content id of ground truth pool to use for celebrity detection.",
        },
    )
    content_type = fields.Str(
        load_default="video",
        metadata={
            "description": (
                '"image" or "video". Image mode optimizes the accuracy for static images.'
            ),
            "enum": ["image", "video"],
        },
    )


class ParamsOcrSchema(Schema):
    """`model: "ocr"`"""
    fps = fields.Int(
        required=True,
        metadata={"description": "Frames per second to sample for text detection.", "example": 1},
    )
    allow_single_frame = fields.Bool(
        load_default=False,
        metadata={
            "description": (
                "Whether to generate a video tag from a single-frame detection (same as celeb)."
            ),
        },
    )
    w_thres = fields.Float(
        required=True,
        metadata={"description": "Word-level confidence threshold.", "example": 0.5},
    )
    l_thres = fields.Bool(
        required=True,
        metadata={"description": "Line-level threshold.", "example": True},
    )


class ParamsLogoSchema(Schema):
    """`model: "logo"`"""
    fps = fields.Int(
        required=True,
        metadata={"description": "Frames per second to sample for logo detection.", "example": 1},
    )


class ParamsCaptionSchema(Schema):
    """`model: "caption"`"""
    fps = fields.Int(
        required=True,
        metadata={"description": "Frames per second to sample for captioning.", "example": 1},
    )


class ParamsSceneDescriptionSchema(Schema):
    """`model: "scene_description"`"""
    num_frames = fields.Int(
        load_default=5,
        metadata={
            "description": (
                "How many frames to sample for each input video chunk. NOTE: often it's most "
                'sensible to run scene description alongside the "tag-aligned" scope.'
            ),
        },
    )
    prompt = fields.Str(
        load_default="Describe this scene.",
        metadata={"description": "Prompt used to describe the scene."},
    )
    model = fields.Str(
        load_default="Qwen/Qwen3-VL-8B-Instruct",
        metadata={
            "description": (
                "Model name to run. Currently this is limited to huggingface model ids in the "
                "Qwen3-VL family."
            ),
        },
    )
    rescale_width = fields.Int(
        load_default=320,
        metadata={
            "description": (
                "Rescale the video to this width before processing. Lowering this value can "
                "improve inference performance, potentially at the cost of model accuracy."
            ),
        },
    )
    rescale_height = fields.Int(
        load_default=240,
        metadata={"description": "Rescale the video to this height before processing."},
    )


# Maps the `model` discriminator to its params schema. Registered in the OpenAPI
# spec in server.py. Models not listed here take no documented parameters.
MODEL_PARAM_SCHEMAS = {
    "asr": ParamsAsrSchema,
    "euro_asr": ParamsMultilingualAsrSchema,
    "llava": ParamsLlavaSchema,
    "celeb": ParamsCelebSchema,
    "ocr": ParamsOcrSchema,
    "logo": ParamsLogoSchema,
    "caption": ParamsCaptionSchema,
    "scene_description": ParamsSceneDescriptionSchema,
}


def model_params_component_name(model: str) -> str:
    """OpenAPI component name for a model's params, e.g. "asr" -> "ModelParamsAsr"."""
    return "ModelParams" + "".join(part.capitalize() for part in model.replace("-", "_").split("_"))


def model_params_oneof_metadata() -> dict:
    """marshmallow field metadata that renders `model_params` as a `oneOf`.

    apispec merges these keys into the field's OpenAPI schema, so `model_params`
    points explicitly at the per-model components registered in the spec. The field
    stays a raw dict at runtime -- see JobSpecSchema.model_params.
    """
    return {
        "description": (
            "Model-specific parameters, selected by the sibling `model` field. The shape "
            "is one of the per-model variants below; models without documented parameters "
            "accept an empty object."
        ),
        "oneOf": [
            {"$ref": f"#/components/schemas/{model_params_component_name(m)}"}
            for m in MODEL_PARAM_SCHEMAS
        ],
    }
