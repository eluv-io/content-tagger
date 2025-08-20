from dataclasses import dataclass

@dataclass
class FetcherConfig:
    max_downloads: int
    parts_path: str

@dataclass
class Source:
    name: str
    filepath: str
    offset: float | None

@dataclass
class DownloadResult:
    successful_sources: list[Source]
    failed_part_hashes: list[str]

@dataclass
class VodDownloadRequest:
    stream_name: str
    start_time: int | None
    end_time: int | None
    replace: bool

@dataclass
class StreamMetadata:
    parts: list[str]
    part_duration: float
    fps: float | None
    codec_type: str