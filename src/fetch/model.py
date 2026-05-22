from dataclasses import dataclass
from typing import Protocol

@dataclass
class FetcherConfig:
    author: str
    max_downloads: int

@dataclass
class Source:
    name: str
    filepath: str
    # relative to media (in ms)
    offset: int
    # absolute unix time (in ms)
    wall_clock: int | None

class Scope:
    def get_stream(self) -> str:
        if hasattr(self, "stream"):
            return getattr(self, "stream")
        return ""

@dataclass
class TimeRangeScope(Scope):
    start_time: int | None = 0
    end_time: int | None = int(1e10)
    chunk_size: int = 600
    stream: str = ""
    type: str = "processor"

@dataclass
class AssetScope(Scope):
    assets: list[str] | None = None
    type: str = "assets"
    
    def get_stream(self) -> str:
        return "assets"

@dataclass
class VideoScope(Scope):
    stream: str = ""
    # in seconds
    start_time: int = 0
    end_time: int = int(1e10)
    type: str = "video"

@dataclass
class LiveScope(Scope):
    stream: str = ""
    segment_length: int = 4
    max_duration: int | None = None
    type: str = "livestream"

@dataclass
class TagAlignedScope(Scope):
    stream: str = ""
    start_time: int = 0
    end_time: int = int(1e10)
    track: str = "shot_detection"
    type: str = "tag-aligned"

@dataclass
class VideoMetadata:
    parts: list[str]
    fps: float | None
    codec_type: str
    part_duration: float

@dataclass
class MediaMetadata:
    sources: list[str]
    fps: float | None

@dataclass
class DownloadRequest:
    output_dir: str
    scope: Scope
    # list of sources to ignore (for diff-based tagging)
    ignore_sources: list[str]
    
@dataclass
class DownloadResult:
    sources: list[Source]
    failed: list[str]
    # so the tagger knows when to stop the job
    done: bool

class FetchSession(Protocol):
    """"""

    def metadata(self) -> MediaMetadata:
        """Get the media metadata for the content being downloaded. This is useful for the uploader to know how to calculate the offsets of each source."""
        ...

    def download(self) -> DownloadResult:
        """Download a batch of content. Can be called multiple times and will specify done=True in the result when there is no more content to download.
        
        Downloading in batches helps us to enable the tagging to begin before the entire content is downloaded, which is especially important for live content.
        """
        ...

    @property
    def path(self) -> str:
        """Path to the downloaded content

        This is so that we know what directory to mount when starting a container.

        # TODO: we can consider having it return the original DownloadRequest instead
        """
        ...