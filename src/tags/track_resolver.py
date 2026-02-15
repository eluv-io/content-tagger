
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

    def resolve(self, model_name: str) -> TrackArgs:
        if model_name in self.cfg.mapping:
            return deepcopy(self.cfg.mapping[model_name])
        else:
            return self._default_track_args(model_name)
        
    def _default_track_args(self, model_name: str) -> TrackArgs:
        return TrackArgs(name=model_name, label=model_name.replace("_", " ").title())