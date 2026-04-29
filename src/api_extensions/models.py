from dataclasses import dataclass

from src.tag_containers.registry import ContainerRegistry
from src.tags.track_resolver import TrackArgs, TrackResolver

@dataclass
class ModelSpec:
    name: str
    description: str
    type: str
    tag_tracks: list[TrackArgs]
    dependencies: list[str]


@dataclass
class ListingResponse:
    models: list[ModelSpec]


def list_models(
    registry: ContainerRegistry,
    track_resolver: TrackResolver                
) -> ListingResponse:
    models = registry.services()
    specs = []
    for m in models:
        cfg = registry.get_model_config(m)
        if not cfg.description:
            # hide models without description from public listing to be used for internal purposes
            continue
        track = track_resolver.resolve(m)
        specs.append(
            ModelSpec(
                name=m,
                description=cfg.description,
                type=cfg.type,
                tag_tracks=[track],
                dependencies=cfg.dependencies
            )
        )
    return ListingResponse(
        models=specs
    )