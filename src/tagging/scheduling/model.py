from typing import Literal
from dataclasses import dataclass
import time

from src.common.model import SystemResources


JobStateDescription = Literal["Queued", "Running", "Completed", "Failed", "Stopped"]

@dataclass
class ContainerJobStatus:
    status: JobStateDescription
    time_started: float
    time_ended: float | None
    error: Exception | None

    @staticmethod
    def starting() -> 'ContainerJobStatus':
        return ContainerJobStatus(
            status="Queued",
            time_started=time.time(),
            time_ended=None,
            error=None
        )

@dataclass 
class SysConfig:
    # map gpu idx -> gpu type
    gpus: list[str]
    # arbitrary key-value pairs for other resources
    resources: SystemResources