
from copy import deepcopy
from dataclasses import dataclass, field

@dataclass(frozen=True)
class TrackArgs:
    name: str
    label: str

@dataclass(frozen=True)
class TrackResolverConfig:
    mapping: dict[str, TrackArgs]

class TrackResolver:
    def __init__(self, cfg: TrackResolverConfig):
        self.cfg = cfg
        self._reverse_mapping: dict[str, str] = {
            track_args.name: model_name
            for model_name, track_args in cfg.mapping.items()
        }

    def resolve(self, model_name: str) -> TrackArgs:
        if model_name in self.cfg.mapping:
            return deepcopy(self.cfg.mapping[model_name])
        else:
            return self._default_track_args(model_name)

    def reverse_resolve(self, track_name: str) -> str:
        """Resolve a track name back to the original model name."""
        return self._reverse_mapping.get(track_name, track_name)
        
    def _default_track_args(self, model_name: str) -> TrackArgs:
        return TrackArgs(name=model_name, label=model_name.replace("_", " ").title())