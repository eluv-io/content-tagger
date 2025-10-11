import threading

from src.tagging.fabric_tagging.model import *
from src.tag_containers.containers import TagContainer
from src.fetch.model import *

@dataclass
class MediaState:
    downloaded: list[Source]
    worker: DownloadWorker

@dataclass
class JobState:
    """Mutable state of the tagging job."""
    status: JobStatus
    taghandle: str
    uploaded_sources: list[str]
    message: str
    media: MediaState
    upload_job: str
    container: TagContainer | None
    tagging_done: threading.Event | None

@dataclass
class TagJob:
    """Context of the tagging job."""
    args: JobArgs
    state: JobState
    stop_event: threading.Event

    def get_id(self) -> JobID:
        if isinstance(self.args.scope, AssetScope):
            stream = "assets"
        elif isinstance(self.args.scope, VideoScope):
            stream = self.args.scope.stream
        else:
            raise ValueError(f"unknown scope type: {type(self.args.scope)}")
        return JobID(qhit=self.args.q.qhit, feature=self.args.feature, stream=stream)

@dataclass
class JobStore:
    active_jobs: dict[JobID, TagJob] = field(default_factory=dict)
    inactive_jobs: dict[JobID, TagJob] = field(default_factory=dict)