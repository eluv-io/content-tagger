
from enum import Enum
import threading
import queue
from dataclasses import dataclass

from src.tagging.scheduling.model import *
from src.tag_containers.containers import TagContainer


class MessageType(Enum):
    START_JOB = "start_job"
    STOP_JOB = "stop_job"
    CHECK_CAPACITY = "check_capacity"
    GET_STATUS = "get_status"
    SHUTDOWN = "shutdown"
    CONTAINER_FINISHED = "container_finished"

@dataclass
class Message:
    type: MessageType
    data: dict
    response_queue: queue.Queue | None = None

@dataclass
class StartJobRequest:
    container: TagContainer
    finished: threading.Event | None = None

@dataclass
class StopJobRequest:
    jobid: str
    status: JobStateDescription
    error: Exception | None = None