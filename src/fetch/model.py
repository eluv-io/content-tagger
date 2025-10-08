from dataclasses import dataclass

from src.common.errors import BadRequestError

@dataclass
class FetcherConfig:
    max_downloads: int
    parts_dir: str
    author: str

@dataclass
class Source:
    name: str
    filepath: str
    offset: float

@dataclass
class AssetScope:
    assets: list[str] | None

@dataclass
class VideoScope:
    start_time: int = 0
    end_time: float | int = float("inf")

@dataclass
class DownloadRequest:
    # asset is a special case
    stream_name: str
    preserve_track: str
    scope: AssetScope | VideoScope

    def __post_init__(self):
        if self.stream_name == "assets" and not isinstance(self.scope, AssetScope):
            raise BadRequestError("Invalid scope type for assets stream")

    def __str__(self):
        return f"DownloadRequest(stream_name={self.stream_name}, preserve_track={self.preserve_track}, scope={self.scope})"

@dataclass
class StreamMetadata:
    parts: list[str]
    part_duration: float
    fps: float | None
    codec_type: str

@dataclass
class DownloadResult:
    successful_sources: list[Source]
    failed: list[str]
    stream_meta: StreamMetadata | None