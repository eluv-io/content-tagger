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
    offset: float

class Scope: ...

@dataclass
class AssetScope(Scope):
    assets: list[str] | None

@dataclass
class VideoScope(Scope):
    stream: str
    start_time: int
    end_time: float | int

@dataclass
class LiveScope(Scope):
    stream: str
    chunk_size: int

class MediaMetadata: ...

@dataclass
class VideoMetadata(MediaMetadata):
    parts: list[str]
    part_duration: float
    fps: float | None
    codec_type: str

@dataclass
class AssetMetadata(MediaMetadata): ...

@dataclass
class DownloadRequest:
    # used to avoid re-downloading parts if this track has already been tagged
    # TODO: this logic doesn't belong in the fetcher, we should instead pass in 
    # the sources to not download from the tagger.
    preserve_track: str
    output_dir: str
    scope: Scope

    def __str__(self):
        return f"DownloadRequest(preserve_track={self.preserve_track}, scope={self.scope})"
    
@dataclass
class DownloadResult:
    sources: list[Source]
    failed: list[str]
    # so the tagger knows when to stop the job
    done: bool

class DownloadWorker(Protocol):
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