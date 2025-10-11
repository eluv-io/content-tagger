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
    # TODO: this logic might not belong in the fetcher, maybe we can instead pass in 
    # the sources to not download. 
    preserve_track: str
    output_dir: str
    scope: Scope

    def __str__(self):
        return f"DownloadRequest(preserve_track={self.preserve_track}, scope={self.scope})"
    
@dataclass
class DownloadResult:
    sources: list[Source]
    failed: list[str]
    done: bool

class DownloadWorker(Protocol):

    def metadata(self) -> MediaMetadata:
        ...

    def download(self) -> DownloadResult:
        ...

    @property
    def path(self) -> str:
        ...