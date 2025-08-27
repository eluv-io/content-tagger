
from dataclasses import dataclass

## Schema

@dataclass
class Tag:
    start_time: int
    end_time: int
    text: str
    additional_info: dict
    source: str
    jobid: str

@dataclass
class UploadJob:
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