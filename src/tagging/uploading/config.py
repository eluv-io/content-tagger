from dataclasses import dataclass

@dataclass(frozen=True)
class TrackArgs:
    name: str
    label: str

@dataclass(frozen=True)
class UploaderConfig:
    # map feature id to TrackArgs
    track_mapping: dict[str, TrackArgs]