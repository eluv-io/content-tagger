from dataclasses import dataclass


@dataclass(frozen=True)
class ModelStatusSummary:
    model: str
    track: str
    last_run: float
    percent_completion: float


@dataclass(frozen=True)
class ContentStatusResponse:
    models: list[ModelStatusSummary]
