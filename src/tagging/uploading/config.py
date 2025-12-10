from dataclasses import dataclass, field

@dataclass(frozen=True)
class TrackArgs:
    name: str
    label: str

@dataclass(frozen=True)
class ModelUploadArgs:
    default: TrackArgs
    overrides: dict[str, TrackArgs] = field(default_factory=dict)

@dataclass(frozen=True)
class UploaderConfig:
    # map feature name to ModelUploadArgs
    model_params: dict[str, ModelUploadArgs]