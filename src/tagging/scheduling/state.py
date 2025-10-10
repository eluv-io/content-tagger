from dataclasses import dataclass
import threading

from src.tagging.scheduling.model import *
from src.tag_containers.containers import TagContainer

@dataclass
class ContainerJob:
    container: TagContainer
    jobstatus: ContainerJobStatus
    gpus_used: list[int]
    # trigger downstream tasks
    finished: threading.Event | None

@dataclass
class ResourceState:
    total: SystemResources
    available: SystemResources
    gpu_status: list[bool]