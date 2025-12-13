import math

from src.tags.tagstore.model import Tag
from src.tags.tagstore.abstract import Tagstore
from src.tag_containers.model import ModelTag
from src.tagging.uploading.config import UploaderConfig, TrackArgs
from src.fetch.model import VideoMetadata
from src.common.content import Content
from src.common.logging import logger
from src.tagging.fabric_tagging.media_state import MediaState

class UploadSession:
    
    def __init__(
        self,
        feature: str,
        media_state: MediaState,
        config: UploaderConfig,
        tagstore: Tagstore,
        source_q: Content,
        destination_qid: str,
    ):

        self.feature = feature
        self.media = media_state  # Read-only reference to job's media state
        self.config = config
        self.tagstore = tagstore
        self.source_q = source_q

        self.dest_q = self._resolve_destination(source_q, destination_qid)
        
        # Mutable state
        self.track_to_batch: dict[str, str] = {}
        self.uploaded_tags: set[ModelTag] = set()
        self.uploaded_sources: set[str] = set()

    def upload_tags(self, tags: list[ModelTag], retry: bool) -> None:
        """Main upload method - formats and uploads tags to tagstore"""
        media_to_source = {s.filepath: s for s in self.media.downloaded}

        new_outputs = [t for t in tags if t.source_media in media_to_source and t not in self.uploaded_tags]

        if not new_outputs:
            return

        logger.info(
            "uploading new tags", 
            extra={
                "num_tags": len(new_outputs),
                "feature": self.feature,
                "source_qid": self.source_q.qid,
            }
        )

        stream_meta = self.media.worker.metadata()
        fps = None
        if isinstance(stream_meta, VideoMetadata):
            # in order to map the frame tags to their appropriate timestamps
            # TODO: missing this for live
            fps = stream_meta.fps

        # convert ModelTag to Tagstore DTO
        tags2upload: list[Tag] = []

        for model_tag in new_outputs:
            original_src = media_to_source[model_tag.source_media]
            tag = Tag(
                start_time=model_tag.start_time,
                end_time=model_tag.end_time,
                text=model_tag.text,
                additional_info={},
                frame_tags=model_tag.frame_tags,
                source=original_src.name,
                batch_id=self._get_batch(model_tag.track),
            )
            tags2upload.append(self._fix_tag_timing_info(tag, original_src.offset, original_src.wall_clock, fps))

        try:
            self._post_tags(tags2upload, q=self.dest_q)
        except Exception as e:
            if retry:
                logger.opt(exception=e).error("error uploading tags, but retry is set to true, will retry on next upload tick", extra={"destination qid": self.dest_q.qid, "feature": self.feature})
            else:
                raise

        self.uploaded_sources.update(media_to_source[out.source_media].name for out in new_outputs)
        self.uploaded_tags.update(new_outputs)
    
    def _resolve_destination(self, source_q: Content, destination_qid: str) -> Content:
        """Resolve destination content object"""
        if destination_qid == "" or destination_qid == source_q.qid:
            return source_q
        return source_q.get_child(destination_qid)
    
    def _fix_tag_timing_info(self, tag: Tag, offset: int, wall_clock: int | None, fps: float | None) -> Tag:
        """Fix tag timestamps & frame indices. The model outputs timestamps relative to the start of the media file
        but this will help place it relative to the full content object (or do nothing for assets).
        """
        if wall_clock is not None:
            tag.additional_info["timestamp_ms"] = wall_clock + tag.start_time
        tag.start_time += offset
        tag.end_time += offset
        if tag.frame_tags:
            if fps is not None:
                tag = self._fix_frame_indices(tag, offset, fps)
            else:
                logger.warning("model returned frame tags, but stream fps is unknown: removing frame tags.")
                tag.frame_tags = {}
        return tag

    def _fix_frame_indices(self, tag: Tag, offset: float, fps: float) -> Tag:
        if not tag.frame_tags:
            return tag

        frame_tags = tag.frame_tags
        frame_offset = round(offset * fps)
        residual = (offset * fps) - frame_offset
        if not math.isclose(residual, 0.0, abs_tol=1e-6):
            logger.warning(f"Non-integer frame offset detected\noffset: {offset}, fps: {fps}, frame_offset: {frame_offset}, residual: {residual}")

        adjusted = {}
        for frame_idx, label in frame_tags.items():
            try:
                frame_idx = int(frame_idx)
            except ValueError:
                logger.error(f"Invalid frame index: {tag}")
                continue
            adjusted[frame_idx + frame_offset] = label

        tag.frame_tags = adjusted
        return tag
    
    def _get_destination_q(self, q: Content, destination_qid: str) -> Content:
        if destination_qid == q.qid:
            return q

        dest_q = q.get_child(destination_qid)
        return dest_q
    
    def _get_batch(self, track: str) -> str:
        if not track:
            track = self._get_default_track(self.feature)

        if track in self.track_to_batch:
            return self.track_to_batch[track]

        track_args = self._get_override_track(self.feature, track)

        try:
            self.tagstore.create_track(
                qhit=self.dest_q.qid,
                name=track_args.name,
                label=track_args.label,
                q=self.dest_q,
            )
        except Exception:
            # track may already exist
            pass

        db_track = self.tagstore.get_track(
            qhit=self.dest_q.qid,
            name=track_args.name,
            q=self.dest_q,
        )

        assert db_track is not None and db_track.name == track_args.name

        ts_batch = self.tagstore.create_batch(
            qhit=self.dest_q.qid,
            track=db_track.name,
            author="tagger",
            q=self.dest_q,
        )

        self.track_to_batch[track] = ts_batch.id
        return self.track_to_batch[track]

    def _post_tags(self, tags: list[Tag], q: Content) -> None:
        """Upload tags to tagstore (called from actor thread)"""
        if not tags:
            return
        
        # group by batch
        batch_to_tags = {}
        for tag in tags:
            batch_to_tags.setdefault(tag.batch_id, []).append(tag)

        logger.info("uploading tags", extra={"num_tags": len(tags), "qhit": q.qhit, "num_batches": len(batch_to_tags)})

        for batch, tags in batch_to_tags.items():
            try:
                self.tagstore.upload_tags(tags, batch, q=q)
            except Exception as e:
                logger.opt(exception=e).error("error uploading tags", extra={"destination qid": q.qid})
                raise

    def _get_default_track(self, feature: str) -> str:
        return self.config.model_params[feature].default.name
    
    def _get_override_track(self, feature: str, track: str) -> TrackArgs:
        overrides = self.config.model_params[feature].overrides
        if track in overrides:
            return overrides[track]
        return TrackArgs(
            name=track,
            label=track.replace("_", " ").title()
        )