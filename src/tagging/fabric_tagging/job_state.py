import threading
from dataclasses import dataclass, field

from src.tagging.fabric_tagging.model import *
from src.tag_containers.containers import TagContainer
from src.fetch.model import *
from src.tagging.uploading.uploader import UploadSession
from src.tagging.fabric_tagging.media_state import MediaState

@dataclass
class JobState:
    """Mutable state of the tagging job."""
    status: JobStateDescription
    # internal handle used in the ContainerScheduler to identify the job
    taghandle: str
    media: MediaState
    upload_session: UploadSession
    container: TagContainer | None
    # callers can pass an event to be notified when tagging is done
    tagging_done: threading.Event
    fetch_retry_count: int
    warnings: list[str]
    time_started: float
    time_ended: float | None

    @staticmethod
    def starting(media: MediaState, upload_session: UploadSession) -> "JobState":
        """Create a JobState in starting state."""
        return JobState(
            status="Queued",
            taghandle="",
            media=media,
            upload_session=upload_session,
            time_started=time.time(),
            time_ended=None,
            tagging_done=threading.Event(),
            container=None,
            fetch_retry_count=0,
            warnings=[],
        )

@dataclass
class TagJob:
    """Context of the tagging job."""
    args: JobArgs
    state: JobState
    # asynchronous stop event to signal the job to stop
    stop_event: threading.Event

    def get_id(self) -> JobID:
        """Get a human-readable identifier for the job (qid, feature, stream)."""
        if isinstance(self.args.scope, AssetScope):
            stream = "assets"
        elif isinstance(self.args.scope, VideoScope):
            stream = self.args.scope.stream
        elif isinstance(self.args.scope, LiveScope):
            stream = "video"
        elif isinstance(self.args.scope, TimeRangeScope):
            stream = "video"
        else:
            raise ValueError(f"unknown scope type: {type(self.args.scope)}")
        return JobID(qid=self.args.q.qid, feature=self.args.feature, stream=stream)

@dataclass
class JobStore:
    """Store all tagging jobs."""
    active_jobs: dict[JobID, TagJob] = field(default_factory=dict)
    inactive_jobs: dict[JobID, TagJob] = field(default_factory=dict)