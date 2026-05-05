from dataclasses import dataclass

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