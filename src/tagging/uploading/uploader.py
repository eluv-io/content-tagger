
from dataclasses import asdict

from src.common.errors import BadRequestError
from src.common.logging.timing import timeit
from src.tags.datastore.model import Tag
from src.tags.datastore.abstract import Datastore
from src.tag_containers.model import ModelTag
from src.common.content import Content
from src.common.logging import logger
from src.tagging.fabric_tagging.model import TagContentStatusReport
from src.tags.track_resolver import TrackArgs, TrackResolver

class UploadSession:
    def __init__(
        self,
        feature: str,
        track_resolver: TrackResolver,
        datastore: Datastore,
        track_suffix: str,
        dest_q: Content,
        hidden: bool = False,
    ):

        self.feature = feature
        self.track_resolver = track_resolver
        self.datastore = datastore
        self.dest_q = dest_q
        self.track_suffix = track_suffix
        self.hidden = hidden
        # Mutable state
        # a single batch spans the whole session (one model run over many tracks)
        self.batch_id: str | None = None
        self.created_tracks: set[str] = set()
        self.uploaded_tags: set[ModelTag] = set()
        # every source we've seen this session, from progress (tagged_sources) or from
        # the tags themselves, used to delete pre-existing tags for those sources
        self.seen_sources: set[str] = set()
        # sources whose pre-existing tags have already been deleted, so we delete each
        # at most once (before its first post)
        self.deleted_sources: set[str] = set()

    def upload_tags(
        self, 
        tags: list[ModelTag],
        tagged_sources: list[str]
    ) -> None:
        """Main upload method - formats and uploads tags to the datastore"""
        self.seen_sources.update(tagged_sources)
        self.seen_sources.update(t.source_media for t in tags)

        with timeit("deduplicating tags", min_duration=1):
            new_inputs = [t for t in tags if t not in self.uploaded_tags]

        if new_inputs:
            logger.info(
                "uploading new tags",
                num_new_tags=len(new_inputs),
                feature=self.feature,
                dest_qid=self.dest_q.qid,
            )

        # Build tags grouped by their (suffixed) track. A single batch spans all
        # tracks; the track is supplied per upload.
        tags_by_track: dict[str, list[Tag]] = {}
        if new_inputs:
            batch_id = self._get_or_create_batch()
            for t in new_inputs:
                track = self._ensure_track(t.model_track)
                tags_by_track.setdefault(track, []).append(
                    Tag(
                        # empty -> not created yet in live tagstore
                        id="",
                        start_time=t.start_time,
                        end_time=t.end_time,
                        data=t.data,
                        additional_info=t.additional_info,
                        source=t.source_media,
                        batch_id=batch_id,
                        frame_info=t.frame_info,
                    )
                )

        # Delete pre-existing tags for this model's newly-seen sources so a re-run
        # replaces rather than duplicates. Deletion is scoped to the model, which
        # covers every track the model produces in one call.
        sources_to_delete = self.seen_sources - self.deleted_sources

        # a failure leaves the session's state untouched so the caller can retry
        if sources_to_delete:
            self.datastore.delete_tags_by_source(
                sources=list(sources_to_delete),
                model=self._model_name(),
                q=self.dest_q,
            )
        self._post_tags(tags_by_track, q=self.dest_q)

        self.deleted_sources.update(sources_to_delete)

        self.uploaded_tags.update(new_inputs)

    def upload_report(self, report: TagContentStatusReport) -> None:
        """Upload a tagging report to the datastore, recorded on the session's batch."""
        batch = self._get_or_create_batch()
        self.datastore.update_batch(batch_id=batch, additional_info={"tagger": asdict(report)}, q=self.dest_q)

    def has_batch(self) -> bool:
        """Whether this session has written anything to its store yet."""
        return self.batch_id is not None

    def _model_name(self) -> str:
        """The model identifier recorded on the session's batch (and used to scope
        source deletions). The track suffix keeps distinct runs from colliding."""
        return self.track_resolver.apply_suffix(self.feature, self.track_suffix)

    def _get_or_create_batch(self) -> str:
        """Return the single batch for this session, creating it on first use."""
        if self.batch_id is None:
            ts_batch = self.datastore.create_batch(
                model=self._model_name(),
                author="tagger",
                q=self.dest_q,
            )
            self.batch_id = ts_batch.id
        return self.batch_id

    def _ensure_track(self, model_track: str) -> str:
        """Ensure the (suffixed) track for a model_track exists and return its name."""
        if model_track:
            # get the label
            label = self.track_resolver.get_label(model_track)
            track_args = TrackArgs(name=model_track, label=label)
        else:
            track_args = self.track_resolver.resolve(self.feature)[0]

        track = self.track_resolver.apply_suffix(track_args.name, self.track_suffix)
        label = track_args.label

        if self.track_suffix:
            label += f" {self.track_suffix}"

        if track in self.created_tracks:
            return track

        additional_info = {"hidden": True} if self.hidden else None

        try:
            self.datastore.create_track(
                name=track,
                label=label,
                q=self.dest_q,
                additional_info=additional_info,
            )
        except Exception:
            # track may already exist
            pass

        db_track = self.datastore.get_track(
            name=track,
            q=self.dest_q,
        )

        assert db_track is not None and db_track.name == track
        self.created_tracks.add(track)
        return track

    def _post_tags(self, tags_by_track: dict[str, list[Tag]], q: Content) -> None:
        """Upload tags to the datastore, grouped by track under the session's batch."""
        total = sum(len(tags) for tags in tags_by_track.values())
        if not total:
            return

        batch_id = self._get_or_create_batch()

        logger.info("uploading tags", num_tags=total, qid=q.qid, num_tracks=len(tags_by_track))

        for track, tags in tags_by_track.items():
            try:
                self._upload_tags_with_batch(batch_id, tags, track, q)
            except Exception as e:
                logger.opt(exception=e).error("error uploading tags", destination_qid=q.qid)
                raise

    def _upload_tags_with_batch(self, batch_id: str, tags: list[Tag], track: str, q: Content) -> None:
        """
        Uploads a list of tags to the datastore under the specified batch ID and track.
        Uploads in small increments to avoid sending too many tags in a single request.
        """
        chunk_size = 5000
        for i in range(0, len(tags), chunk_size):
            chunk = tags[i:i + chunk_size]
            try:
                self.datastore.upload_tags(chunk, batch_id, track, q=q)
            except Exception as e:
                logger.opt(exception=e).error("error uploading tags chunk", destination_qid=q.qid, batch_id=batch_id)
                raise


class Uploader:
    """Routes one model run's outputs to the stores that can hold them.

    Text tags go to the tagstore and vectors to the vectorstore, each under its own
    `UploadSession` (and so its own batch). A model that emits both writes to both.
    """

    def __init__(
        self,
        feature: str,
        track_resolver: TrackResolver,
        tagstore: Datastore,
        vectorstore: Datastore | None,
        track_suffix: str,
        dest_q: Content,
        do_retry: bool,
        hidden: bool = False,
    ):
        self.feature = feature
        self.dest_q = dest_q
        self.retry = do_retry

        def session(store: Datastore) -> UploadSession:
            return UploadSession(
                feature=feature,
                track_resolver=track_resolver,
                datastore=store,
                track_suffix=track_suffix,
                dest_q=dest_q,
                hidden=hidden,
            )

        self.tag_session = session(tagstore)
        self.vector_session = session(vectorstore) if vectorstore is not None else None
        self.uploaded_sources: set[str] = set()

    def upload(
        self,
        tags: list[ModelTag],
        tagged_sources: list[str]
    ) -> None:
        """Dispatch the model's outputs to the tagstore, the vectorstore, or both."""
        vectors = [t for t in tags if t.is_vector()]
        texts = [t for t in tags if not t.is_vector()]

        if vectors and self.vector_session is None:
            raise BadRequestError(
                f"model {self.feature} produced vectors but no index_qid was given: "
                "specify index_qid in the tagger options to choose the vector index to write to"
            )

        try:
            self.tag_session.upload_tags(texts, tagged_sources)
            if self.vector_session is not None:
                self.vector_session.upload_tags(vectors, tagged_sources)
        except Exception as e:
            if not self.retry:
                raise
            logger.opt(exception=e).error(
                "error uploading tags, but retry is set to true, will retry on next upload tick",
                destination_qid=self.dest_q.qid,
                feature=self.feature,
            )
            return

        self.uploaded_sources.update(tagged_sources)

    def upload_report(self, report: TagContentStatusReport) -> None:
        """Record the run's report.

        The tagstore is the system of record - every run reports there. But the vectorstore also gets a copy.
        """
        self.tag_session.upload_report(report)
        if self.vector_session is not None and self.vector_session.has_batch():
            self.vector_session.upload_report(report)

    def get_uploaded_sources(self) -> list[str]:
        """Get the set of source media that have been tagged in this run."""
        return list(self.uploaded_sources)
