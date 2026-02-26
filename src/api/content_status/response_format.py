from dataclasses import dataclass


@dataclass(frozen=True)
class ModelStatusSummary:
    model: str
    track: str
    last_run: str
    percent_completion: float


@dataclass(frozen=True)
class ContentStatusResponse:
    models: list[ModelStatusSummary]
