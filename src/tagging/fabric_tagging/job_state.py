import threading

from src.tagging.fabric_tagging.model import *
from src.tag_containers.containers import TagContainer
from src.fetch.model import Source, DownloadWorker

@dataclass
class MediaState:
    downloaded: list[Source]
    worker: DownloadWorker

@dataclass
class JobState:
    status: JobStatus
    taghandle: str
    uploaded_sources: list[str]
    message: str
    media: MediaState
    container: TagContainer | None

@dataclass
class TagJob:
    args: JobArgs
    state: JobState
    upload_job: str
    media_dir: str
    stop_event: threading.Event
    tagging_done: threading.Event | None

    def get_id(self) -> JobID:
        assert self.args.runconfig.stream is not None
        return JobID(qhit=self.args.q.qhit, feature=self.args.feature, stream=self.args.runconfig.stream)

@dataclass
class JobStore:
    active_jobs: dict[JobID, TagJob] = field(default_factory=dict)
    inactive_jobs: dict[JobID, TagJob] = field(default_factory=dict)