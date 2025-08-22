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

# not part of the db
@dataclass
class FrameTag:
    frame_idx: int
    box: list[int]
    text: str
    confidence: float