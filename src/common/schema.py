from dataclasses import dataclass

# not part of the db
@dataclass
class FrameTag:
    frame_idx: int
    box: list[int]
    text: str
    confidence: float