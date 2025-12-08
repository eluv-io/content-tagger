import threading
from dataclasses import dataclass, field

from src.tagging.fabric_tagging.model import *
from src.tag_containers.containers import TagContainer
from src.tag_containers.model import ModelTag
from src.fetch.model import *

@dataclass
class MediaState:
    # used to keep track of sources that need to be downloaded for status
    # also used to help the uploader know how to compute the offset
    downloaded: list[Source]
    # used to get media
    worker: FetchSession

@dataclass
class JobState:
    """Mutable state of the tagging job."""
    status: JobStatus
    # internal handle used in the ContainerScheduler to identify the job
    taghandle: str
    # used for status reporting
    uploaded_sources: set[str]
    # prevent double upload
    uploaded_tags: set[ModelTag]
    # sources with no corresponding tags
    missing_tags: set[str]
    message: str
    media: MediaState
    # tagstore job/track id to know where to upload tags
    tag_batch: str
    container: TagContainer | None
    # callers can pass an event to be notified when tagging is done
    tagging_done: threading.Event

    @staticmethod
    def starting(batch_id: str, worker: FetchSession) -> "JobState":
        """Create a JobState in starting state."""
        return JobState(
            status=JobStatus.starting(),
            taghandle="",
            uploaded_sources=set(),
            uploaded_tags=set(),
            missing_tags=set(),
            message="",
            media=MediaState(
                downloaded=[],
                worker=worker
            ),
            tag_batch=batch_id,
            tagging_done=threading.Event(),
            container=None
        )

@dataclass
class TagJob:
    """Context of the tagging job."""
    args: JobArgs
    state: JobState
    # asynchronous stop event to signal the job to stop
    stop_event: threading.Event

    def get_id(self) -> JobID:
        """Get the job ID. Used as unique identifier for the job."""
        if isinstance(self.args.scope, AssetScope):
            stream = "assets"
        elif isinstance(self.args.scope, VideoScope):
            stream = self.args.scope.stream
        elif isinstance(self.args.scope, LiveScope):
            stream = "video"
        else:
            raise ValueError(f"unknown scope type: {type(self.args.scope)}")
        return JobID(qhit=self.args.q.qhit, feature=self.args.feature, stream=stream)

@dataclass
class JobStore:
    """Store all tagging jobs."""
    active_jobs: dict[JobID, TagJob] = field(default_factory=dict)
    inactive_jobs: dict[JobID, TagJob] = field(default_factory=dict)