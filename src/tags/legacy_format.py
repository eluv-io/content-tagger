from dataclasses import dataclass
from src.common.schema import Tag, UploadJob

from common_ml.tags import VideoTag, FrameTag, AggTag


# maps frame index to feature to list of FrameTag
Overlay = dict[int, dict[str, list[FrameTag]]]

@dataclass
class TrackCollection:
    tracks: dict[str, list[VideoTag]]
    agg_tracks: dict[str, list[AggTag]]
