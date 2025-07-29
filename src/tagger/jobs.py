from typing import List, Optional, Literal
from dataclasses import dataclass, field
import threading

from src.fabric.content import Content

# RunConfig gives model level tagging params
@dataclass
class RunConfig():
    # model config, used to overwrite the model level config
    model: dict=field(default_factory=dict) 
    # stream name to run the model on, None to use the default stream. "image" is a special case which will tag image assets
    stream: str | None=None

@dataclass
class Job:
    status: Literal["Starting", 
                    "Fetching content",
                    "Waiting to be assigned GPU", 
                    "Waiting for CPU resource", 
                    "Completed", 
                    "Failed", 
                    "Stopped"]
    q: Content
    feature: str
    run_config: RunConfig
    stop_event: threading.Event
    media_files: List[str]
    replace: bool
    time_started: float
    failed: List[str]
    allowed_gpus: List[str]
    allowed_cpus: List[str]
    time_ended: Optional[float]=None
    # tag_job_id is the job id returned by the manager, will be None until the tagging starts (status is "Running")
    tag_job_id: Optional[str]=None
    error: Optional[str]=None
    ## wall clock time of the last time this job was "put back" on the queue
    reput_time: int = 0

@dataclass
class JobsStore:
    """
    A store for jobs, used to keep track of active jobs and their statuses.
    """
    active_jobs: dict[str, dict[tuple[str, str], Job]] = field(default_factory=dict)
    inactive_jobs: dict[str, dict[tuple[str, str], Job]] = field(default_factory=dict)