from typing import Literal
from dataclasses import dataclass
import threading

from src.tag_containers.containers import TagContainer

SystemResources = dict[str, int]

JobState = Literal["Queued", "Running", "Completed", "Failed", "Stopped"]

@dataclass
class JobStatus:
    status: JobState
    time_started: float
    time_ended: float | None
    error: Exception | None

@dataclass
class ContainerJob:
    container: TagContainer
    # resource requirements for the job
    reqs: SystemResources
    jobstatus: JobStatus
    gpus_used: list[int]
    # trigger downstream tasks
    finished: threading.Event | None

@dataclass 
class SysConfig:
    # map gpu idx -> gpu type
    gpus: list[str]
    cpu_juice: int