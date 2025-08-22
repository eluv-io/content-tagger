from dataclasses import dataclass

@dataclass
class Tag:
    start_time: int
    end_time: int
    text: str
    additional_info: dict
    source: str
    jobid: str

@dataclass
class Job:
    id: str
    qhit: str
    stream: str | None
    track: str
    timestamp: float
    author: str
