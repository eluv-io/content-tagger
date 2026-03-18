
from dataclasses import asdict

from src.tags.tagstore.model import Tag
from src.tags.tagstore.abstract import Tagstore
from src.tag_containers.model import ModelTag
from src.common.content import Content
from src.common.logging import logger
from src.tagging.fabric_tagging.media_state import MediaState
from src.tagging.fabric_tagging.model import TagContentStatusReport
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
        self.track_resolver = track_resolver
        self.tagstore = tagstore
        self.source_q = source_q

        self.dest_q = self._resolve_destination(source_q, destination_qid)
        
        # Mutable state
        self.track_to_batch: dict[str, str] = {}
        self.uploaded_tags: set[ModelTag] = set()

    def upload_tags(self, tags: list[ModelTag], retry: bool) -> None:
        """Main upload method - formats and uploads tags to tagstore"""
        new_inputs = [t for t in tags if t not in self.uploaded_tags]

        if not new_inputs:
            return

        logger.info(
            "uploading new tags",
            num_new_tags=len(new_inputs),
            feature=self.feature,
            source_qid=self.source_q.qid,
        )

        tags2upload: list[Tag] = [
            Tag(
                start_time=t.start_time,
                end_time=t.end_time,
                text=t.text,
                additional_info=t.additional_info,
                source=t.source_media,
                batch_id=self._get_or_create_batch(t.model_track),
                frame_info=t.frame_info,
            )
            for t in new_inputs
        ]

        try:
            self._post_tags(tags2upload, q=self.dest_q)
        except Exception as e:
            if retry:
                logger.opt(exception=e).error("error uploading tags, but retry is set to true, will retry on next upload tick", destination_qid=self.dest_q.qid, feature=self.feature)
            else:
                raise

        self.uploaded_tags.update(new_inputs)

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