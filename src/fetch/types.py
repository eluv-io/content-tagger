from dataclasses import dataclass

@dataclass
class FetcherConfig:
    max_downloads: int
    parts_path: str
    author: str

@dataclass
class Source:
    name: str
    filepath: str
    offset: float

@dataclass
class VodDownloadRequest:
    stream_name: str
    start_time: int | None
    end_time: int | None
    preserve_track: str

@dataclass
class StreamMetadata:
    parts: list[str]
    part_duration: float
    fps: float
    codec_type: str

@dataclass
class AssetDownloadRequest:
    assets: list[str] | None
    preserve_track: str

@dataclass
class DownloadResult:
    successful_sources: list[Source]
    failed: list[str]
    stream_meta: StreamMetadata | None