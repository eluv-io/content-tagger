from dataclasses import dataclass

@dataclass(frozen=True)
class TagStatus:
    all_sources: list[str]
    tagged_sources: list[str]

@dataclass(frozen=True)
class ContainerInfo:
    image_name: str
    annotations: dict

@dataclass(frozen=True)
class TagJobStatusReport:
    model_params: dict
    container: ContainerInfo
    status: TagStatus