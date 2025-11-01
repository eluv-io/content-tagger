
from dataclasses import dataclass

## Config

@dataclass
class TagstoreConfig:
    base_dir: str=""
    base_url: str=""
    auth_token: str | None = None

## Schema

@dataclass
class Tag:
    start_time: int
    end_time: int
    text: str
    frame_tags: dict
    additional_info: dict
    source: str
    batch_id: str

@dataclass
class Batch:
    id: str
    qhit: str
    stream: str | None
    track: str
    timestamp: float
    author: str

## Args

@dataclass
class CreateJobArgs:
    qhit: str
    track: str
    stream: str
    author: str