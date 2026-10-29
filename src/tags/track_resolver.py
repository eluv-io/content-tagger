
from copy import deepcopy
from dataclasses import dataclass, field

from src.common.model import ModelConfigStore

@dataclass(frozen=True)
class TrackArgs:
    name: str
    label: str

@dataclass
class LabelResolverConfig:
    mapping: dict[str, str]

class TrackResolver:
    def __init__(self, label_configs: LabelResolverConfig, model_configs: ModelConfigStore):
        self.label_configs = label_configs
        self.model_configs = model_configs

    def resolve(self, model_name: str) -> list[TrackArgs]:
        """Resolve a model name to its track args.
        
        Guaranteed to return list of length >= 1
        """
        forward_mapping = self._forward_mapping()
        if model_name in forward_mapping:
            return forward_mapping[model_name]
        else:
            # If no specific mapping, return default track args
            return [self._default_track_args(model_name)]

    def reverse_resolve(self, track_name: str) -> list[str]:
        """Resolve a track name back to the original model name.
        
        Guaranteed to return list of length >= 1
        """
        reverse_mapping: dict[str, list[str]] = {}
        for model_name, track_args_list in self._forward_mapping().items():
            for track_arg in track_args_list:
                reverse_mapping.setdefault(track_arg.name, []).append(model_name)
        return reverse_mapping.get(track_name, [track_name])

    def apply_suffix(self, base: str, suffix: str) -> str:
        """Append a run's track suffix to a track or model name.
        """
        if suffix:
            return f"{base}_{suffix.replace(' ', '_')}"
        return base

    def get_label(self, track_name: str) -> str:
        if track_name in self.label_configs.mapping:
            return self.label_configs.mapping[track_name]
        else:
            return track_name.replace("_", " ").title()
        
    def _default_track_args(self, model_name: str) -> TrackArgs:
        return TrackArgs(name=model_name, label=model_name.replace("_", " ").title())

    def _forward_mapping(self) -> dict[str, list[TrackArgs]]:
        forward_mapping: dict[str, list[TrackArgs]] = {}
        for model_name, model_cfg in self.model_configs.snapshot().items():
            for track in model_cfg.track_outputs:
                label = self.label_configs.mapping.get(track, track.replace("_", " ").title())
                forward_mapping.setdefault(model_name, []).append(TrackArgs(name=track, label=label))
        return forward_mapping
