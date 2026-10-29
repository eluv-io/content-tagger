
from dataclasses import dataclass

## Schema

# payload of a tag: either text (tagstore) or an embedding (vectorstore)
TagData = str | list[float]

def is_vector(data: TagData) -> bool:
    return not isinstance(data, str)

@dataclass
class Tag:
    # if empty string means it doesn't exist yet
    id: str
    start_time: int
    end_time: int
    data: TagData
    additional_info: dict | None
    source: str
    batch_id: str
    frame_info: dict | None = None

@dataclass
class Batch:
    id: str
    qid: str
    model: str
    timestamp: float
    author: str
    additional_info: dict

@dataclass
class Track:
    qid: str
    name: str
    label: str
    additional_info: dict | None = None
