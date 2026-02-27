from copy import deepcopy
from dataclasses import asdict

from src.tags.tagstore.model import Tag
from src.tags.tagstore.abstract import Tagstore
from src.tag_containers.model import ModelTag
from src.fetch.model import AssetMetadata, VideoMetadata
from src.common.content import Content
from src.common.logging import logger
from src.tagging.fabric_tagging.media_state import MediaState
from src.tagging.fabric_tagging.model import TagContentStatusReport, TagJobStatusReport
from src.tags.track_resolver import TrackResolver

class UploadSession:
    
    def __init__(
        self,
        feature: str,
        media_state: MediaState,
        track_resolver: TrackResolver,
        tagstore: Tagstore,
        source_q: Content,
        destination_qid: str,
    ):

        self.feature = feature
        self.media = media_state  # Read-only reference to job's media state
        self.track_resolver = track_resolver
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
            num_new_tags=len(new_outputs),
            feature=self.feature,
            source_qid=self.source_q.qid,
        )

        stream_meta = self.media.worker.metadata()
        fps = None
        if isinstance(stream_meta, VideoMetadata):
            fps = stream_meta.fps

        # convert ModelTag to Tagstore DTO
        tags2upload: list[Tag] = []

        for model_tag in new_outputs:
            original_src = media_to_source[model_tag.source_media]
            tag = Tag(
                start_time=model_tag.start_time,
                end_time=model_tag.end_time,
                text=model_tag.text,
                additional_info=deepcopy(model_tag.additional_info),
                source=original_src.name,
                batch_id=self._get_or_create_batch(model_tag.model_track),
                frame_info=deepcopy(model_tag.frame_info),
            )
            if tag.frame_info is not None and fps is None and not isinstance(stream_meta, AssetMetadata):
                # drop frame tags if we can't fix the frame index and it's not an asset tag
                continue
            tags2upload.append(self._fix_tag_timing_info(tag, original_src.offset, original_src.wall_clock, fps))

        try:
            self._post_tags(tags2upload, q=self.dest_q)
        except Exception as e:
            if retry:
                logger.opt(exception=e).error("error uploading tags, but retry is set to true, will retry on next upload tick", destination_qid=self.dest_q.qid, feature=self.feature)
            else:
                raise

        self.uploaded_sources.update(media_to_source[out.source_media].name for out in new_outputs)
        self.uploaded_tags.update(new_outputs)

    def upload_report(self, report: TagContentStatusReport) -> None:
        """Upload a tagging report to the tagstore as a tag on the content object."""
        batch = self._get_batch(report.params.feature)
        if batch is None:
            logger.error("no batch found for report, skipping upload", feature=report.params.feature, destination_qid=self.dest_q.qid)
            return

        self.tagstore.update_batch(qhit=self.dest_q.qid, batch_id=batch, additional_info={"tagger": asdict(report)}, q=self.dest_q)
    
    def _resolve_destination(self, source_q: Content, destination_qid: str) -> Content:
        """Resolve destination content object"""
        if destination_qid == "" or destination_qid == source_q.qid:
            return source_q
        return source_q.get_child(destination_qid)
    
    def _fix_tag_timing_info(self, tag: Tag, offset: int, wall_clock: int | None, fps: float | None) -> Tag:
        """Fix tag timestamps & frame index. The model outputs timestamps relative to the start of the media file
        but this will help place it relative to the full content object (or do nothing for assets).
        """
        if wall_clock is not None:
            if tag.additional_info is None:
                tag.additional_info = {}
            tag.additional_info["timestamp_ms"] = wall_clock + tag.start_time
        tag.start_time += offset
        tag.end_time += offset
        if tag.frame_info is not None:
            if fps is not None:
                frame_offset = round(offset * fps)
                tag.frame_info["frame_idx"] = tag.frame_info["frame_idx"] + frame_offset
            else:
                tag.frame_info = None
        return tag
    
    def _get_destination_q(self, q: Content, destination_qid: str) -> Content:
        if destination_qid == q.qid:
            return q

        dest_q = q.get_child(destination_qid)
        return dest_q
    
    def _get_batch(self, model_track: str) -> str | None:
        """Get or create a batch for the given model track."""
        track_args = self.track_resolver.resolve(model_track)
        return self.track_to_batch.get(track_args.name)
    
    def _get_or_create_batch(self, model_track: str) -> str:
        if model_track:
            track_args = self.track_resolver.resolve(model_track)
        else:
            track_args = self.track_resolver.resolve(self.feature)

        track = track_args.name

        if track in self.track_to_batch:
            return self.track_to_batch[track]

        try:
            self.tagstore.create_track(
                qhit=self.dest_q.qid,
                name=track,
                label=track_args.label,
                q=self.dest_q,
            )
        except Exception:
            # track may already exist
            pass

        db_track = self.tagstore.get_track(
            qhit=self.dest_q.qid,
            name=track,
            q=self.dest_q,
        )

        assert db_track is not None and db_track.name == track
        ts_batch = self.tagstore.create_batch(
            qhit=self.dest_q.qid,
            track=track,
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

        logger.info("uploading tags", num_tags=len(tags), qhit=q.qhit, num_batches=len(batch_to_tags))

        for batch, tags in batch_to_tags.items():
            try:
                self.tagstore.upload_tags(tags, batch, q=q)
            except Exception as e:
                logger.opt(exception=e).error("error uploading tags", destination_qid=q.qid)
                raise