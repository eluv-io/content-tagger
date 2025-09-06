from dataclasses import dataclass

@dataclass
class FrameTag:
    frame_idx: str
    confidence: float
    box: dict
    text: str