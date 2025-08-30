from typing import Literal
from dataclasses import dataclass
import threading
import time

from src.common.resources import SystemResources
from src.tag_containers.containers import TagContainer

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
class ContainerJob:
    container: TagContainer
    # resource requirements for the job
    reqs: SystemResources
    jobstatus: ContainerJobStatus
    gpus_used: list[int]
    # trigger downstream tasks
    finished: threading.Event | None

@dataclass 
class SysConfig:
    # map gpu idx -> gpu type
    gpus: list[str]
    cpu_juice: int