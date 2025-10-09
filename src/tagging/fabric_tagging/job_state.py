import threading

from src.tagging.fabric_tagging.model import *
from src.tag_containers.containers import TagContainer
from src.fetch.fetch_content import DownloadResult

@dataclass
class JobState:
    # everything that might change during the job
    status: JobStatus
    taghandle: str
    uploaded_sources: list[str]
    message: str
    media: DownloadResult | None
    container: TagContainer | None

    @staticmethod
    def starting() -> 'JobState':
        return JobState(
            status=JobStatus.starting(),
            taghandle="",
            uploaded_sources=[],
            message="",
            media=None,
            container=None
        )

@dataclass
class TagJob:
    args: JobArgs
    state: JobState
    upload_job: str
    stop_event: threading.Event
    tagging_done: threading.Event | None

    def get_id(self) -> JobID:
        assert self.args.runconfig.stream is not None
        return JobID(qhit=self.args.q.qhit, feature=self.args.feature, stream=self.args.runconfig.stream)

@dataclass
class JobStore:
    active_jobs: dict[JobID, TagJob] = field(default_factory=dict)
    inactive_jobs: dict[JobID, TagJob] = field(default_factory=dict)