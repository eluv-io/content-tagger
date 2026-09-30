from dataclasses import dataclass

from marshmallow import Schema, fields

from src.common.model import ModelConfig

@dataclass
class TrackOutput:
    name: str

@dataclass
class ModelSpec:
    name: str
    description: str
    type: str
    tag_tracks: list[TrackOutput]
    dependencies: list[str]
    category: str
    # JSON schema for model_params, None if the model has no documented params
    params_schema: dict | None


@dataclass
class ListingResponse:
    models: list[ModelSpec]


class TrackOutputSchema(Schema):
    name = fields.Str(
        metadata={
            "description": "Track name as stored in the tag store",
            "example": "celebrity_detection",
        }
    )


class ModelSpecSchema(Schema):
    name = fields.Str(metadata={"description": "Model identifier", "example": "celeb"})
    description = fields.Str(
        metadata={
            "description": "Human readable description of what the model does",
            "example": "Celebrity Identification",
        }
    )
    type = fields.Str(metadata={"description": "Model type", "example": "frame"})
    category = fields.Str(
        metadata={
            "description": "Model category for grouping",
            "example": "Frame Level Detection",
        },
    )
    tag_tracks = fields.List(
        fields.Nested(TrackOutputSchema),
        metadata={"description": "Tag tracks this model writes to"},
    )
    dependencies = fields.List(
        fields.Str(),
        metadata={
            "description": "Tag tracks that must exist before this model can be run"
        },
    )
    params_schema = fields.Dict(
        allow_none=True,
        metadata={
            "description": (
                "OpenAPI schema for the model's `model_params`. Null if the model "
                "has no documented parameters."
            ),
        },
    )


class ListingResponseSchema(Schema):
    models = fields.List(
        fields.Nested(ModelSpecSchema),
        metadata={"description": "List of available models"},
    )


def list_models(
    model_configs: dict[str, ModelConfig],
    params_schemas: dict[str, dict],
) -> ListingResponse:
    specs = []
    for m, cfg in model_configs.items():
        if not cfg.description:
            # hide models without description from public listing to be used for internal purposes
            continue
        specs.append(
            ModelSpec(
                name=m,
                description=cfg.description,
                type=cfg.type,
                # TODO: might break evie
                tag_tracks=[TrackOutput(name=output) for output in cfg.track_outputs],
                dependencies=cfg.track_dependencies,
                category=cfg.category,
                params_schema=params_schemas.get(m),
            )
        )
    return ListingResponse(
        models=specs
    )