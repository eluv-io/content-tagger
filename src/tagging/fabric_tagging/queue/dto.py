
from dataclasses import dataclass
from src.service.model import WarningResponse
from src.tagging.fabric_tagging.queue.model import *


@dataclass
class TagDetailsRaw:
    tag_status: str
    time_running: float
    # between 0 and 1
    progress: float
    # legacy
    tagging_progress: str

    # extra detail
    total_parts: int
    downloaded_parts: int
    tagged_parts: int
    warnings: WarningResponse | None = None
    tagged_duration: float | None = None

    def to_model(self) -> TagDetails:
        return TagDetails(
            tag_status=self.tag_status,
            time_running=self.time_running,
            progress=self.progress,
            tagging_progress=self.tagging_progress,
            total_parts=self.total_parts,
            downloaded_parts=self.downloaded_parts,
            tagged_parts=self.tagged_parts,
            warnings=self.warnings,
            tagged_duration=self.tagged_duration if self.tagged_duration is not None else 0,
        )