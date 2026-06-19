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


class ListingResponseSchema(Schema):
    models = fields.List(
        fields.Nested(ModelSpecSchema),
        metadata={"description": "List of available models"},
    )


def list_models(
    model_configs: dict[str, ModelConfig]      
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
                dependencies=cfg.track_dependencies
            )
        )
    return ListingResponse(
        models=specs
    )