from copy import deepcopy

from src.tag_containers.model import ModelTag, Progress
from src.fetch.model import Source

def adjust_progress_sources(statuses: list[Progress], sources: list[Source]) -> list[Progress]:
    """Map the source_media field (the file-path that the container tagged) back to the original source name.
    """

    source_by_filepath = {s.filepath: s for s in sources}
    adjusted_statuses: list[Progress] = []
    for status in statuses:
        src = source_by_filepath.get(status.source_media)
        if src is None:
            continue

        adjusted_statuses.append(
            Progress(
                source_media=src.name
            )
        )
    return adjusted_statuses

def align_tags(tags: list[ModelTag], sources: list[Source], fps: float | None) -> list[ModelTag]:
    """Align ModelTag timestamps relative to the full content object.

    Filters tags to only those whose source_media matches a known source filepath,
    then adjusts start_time, end_time, frame_info, and additional_info to be
    relative to the full content object rather than the individual media file.
    """

    source_by_filepath = {s.filepath: s for s in sources}
    aligned: list[ModelTag] = []
    for tag in tags:
        src = source_by_filepath.get(tag.source_media)
        if src is None:
            continue

        additional_info = deepcopy(tag.additional_info)
        if src.wall_clock is not None:
            if additional_info is None:
                additional_info = {}
            additional_info["timestamp_ms"] = src.wall_clock + tag.start_time

        frame_info = deepcopy(tag.frame_info)
        if frame_info is not None:
            if fps is not None:
                frame_offset = int((src.offset / 1000) * fps)
                frame_info["frame_idx"] = frame_info["frame_idx"] + frame_offset
            else:
                frame_info = None

        # this step I feel a little weird about having in align_tags
        src_name = src.name

        aligned.append(
            ModelTag(
                start_time=tag.start_time + src.offset,
                end_time=tag.end_time + src.offset,
                text=tag.text,
                additional_info=additional_info,
                source_media=src_name,
                model_track=tag.model_track,
                frame_info=frame_info,
            )
        )
    return aligned