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
    name = fields.Str()


class ModelSpecSchema(Schema):
    name = fields.Str()
    description = fields.Str()
    type = fields.Str()
    tag_tracks = fields.List(fields.Nested(TrackOutputSchema))
    dependencies = fields.List(fields.Str())


class ListingResponseSchema(Schema):
    models = fields.List(fields.Nested(ModelSpecSchema))


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