
from dataclasses import dataclass

## Config

@dataclass
class TagstoreConfig:
    base_dir: str=""
    base_url: str=""
    timeout: int=10
    auth_token: str | None = None

## Schema

@dataclass
class Tag:
    # if empty string means it doesn't exist yet
    id: str
    start_time: int
    end_time: int
    text: str
    additional_info: dict | None
    source: str
    batch_id: str
    frame_info: dict | None = None

@dataclass
class Batch:
    id: str
    qid: str
    track: str
    timestamp: float
    author: str
    additional_info: dict

@dataclass
class Track:
    qid: str
    name: str
    label: str
