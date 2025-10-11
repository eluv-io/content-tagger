
from requests.exceptions import HTTPError
from fractions import Fraction

from common_ml.utils.metrics import timeit

from src.common.logging import logger

from src.common.content import Content
from src.common.errors import MissingResourceError, BadRequestError
from src.fetch.model import *
from src.fetch.coordinator import FetchContext
from src.fetch.cache import cache_by_qhash
from src.fetch.workers import *
from src.tags.tagstore.abstract import Tagstore

logger = logger.bind(name="Fetcher")

class Fetcher:
    def __init__(
        self,
        config: FetcherConfig,
        context: FetchContext,
        ts: Tagstore,
    ):
        self.config = config
        self.ts = ts
        self.ctx = context

    def get_worker(
            self, 
            q: Content, 
            req: DownloadRequest, 
            exit: threading.Event | None = None
    ) -> DownloadWorker:
        with timeit(f"Getting media metadata: qhit={q.qhit}, scope={req.scope}"):
            meta = self._get_metadata(q, req.scope)
        with timeit(f"Getting ignored sources: qhit={q.qhit}, scope={req.scope}, track={req.preserve_track}"):
            ignore_sources = self._get_ignored_sources(q, req.preserve_track, req.scope)
        if isinstance(req.scope, VideoScope):
            assert isinstance(meta, VideoMetadata)
            return VodWorker(
                q=q,
                scope=req.scope,
                context=self.ctx,
                meta=meta,
                ignore_parts=ignore_sources,
                output_dir=req.output_dir,
                exit=exit
            )
        elif isinstance(req.scope, AssetScope):
            assert isinstance(meta, AssetMetadata)
            return AssetWorker(
                q=q,
                scope=req.scope,
                context=self.ctx,
                meta=meta,
                ignore_assets=ignore_sources,
                output_dir=req.output_dir,
                exit=exit
            )
        elif isinstance(req.scope, LiveScope):
            assert isinstance(meta, VideoMetadata)
            return LiveWorker(
                q=q,
                scope=req.scope,
                context=self.ctx,
                meta=meta,
                ignore_parts=[],
                output_dir=req.output_dir,
                exit=exit
            )
        else:
            raise BadRequestError(f"Unknown scope type: {type(req.scope)}")

    def _get_ignored_sources(self, q: Content, preserve_track: str, scope: Scope) -> list[str]:
        if not preserve_track:
            return []
        if isinstance(scope, VideoScope) or isinstance(scope, AssetScope):
            # TODO: could get slow, doesn't work with pagination in case of real tagstore.
            existing_tags = self.ts.find_tags(
                author=self.config.author, 
                qhit=q.qid,
                track=preserve_track,
                q=q
            )
            return list(set(tag.source for tag in existing_tags))
        return []
    
    def _get_metadata(self, q: Content, scope: Scope) -> MediaMetadata:
        if isinstance(scope, VideoScope):
            return self._fetch_stream_metadata(q, scope.stream)
        elif isinstance(scope, AssetScope):
            return AssetMetadata()
        elif isinstance(scope, LiveScope):
            return self._fetch_livestream_metadata(q, scope.stream)
        else:
            raise BadRequestError(f"Unknown scope type: {type(scope)}")

    @cache_by_qhash
    def _fetch_stream_metadata(self, q: Content, stream_name: str) -> VideoMetadata:
        """Fetches metadata for a stream based on content type."""
        if self._is_live(q):
            return self._fetch_livestream_metadata(q, stream_name)

        if self._is_legacy_vod(q):
            return self._fetch_legacy_vod_metadata(q, stream_name)
        else:
            return self._fetch_vod_metadata(q, stream_name)

    def _fetch_vod_metadata(self, q: Content, stream_name: str) -> VideoMetadata:
        """Fetches metadata for modern VOD content."""
        try:
            transcodes = q.content_object_metadata(
                metadata_subtree="transcodes", resolve_links=False
            )
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve transcodes for {q.qhit}") from e

        assert isinstance(transcodes, dict)

        try:
            streams = q.content_object_metadata(
                metadata_subtree="offerings/default/playout/streams",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve streams for {q.qhit}") from e
        
        assert isinstance(streams, dict)

        if stream_name not in streams:
            raise MissingResourceError(f"Stream {stream_name} not found in {q.qhit}")

        representations = streams[stream_name].get("representations", {})

        transcode_id = None
        for repr in representations.values():
            tid = repr["transcode_id"]
            if transcode_id is None:
                transcode_id = tid
                continue
            if tid != transcode_id:
                logger.warning(
                    f"Multiple transcode_ids found for stream {stream_name} in {q.qhit}! Continuing with the first one found."
                )

        if transcode_id is None:
            raise MissingResourceError(
                f"Transcode_id not found for stream {stream_name} in {q.qhit}"
            )

        transcode_meta = transcodes[transcode_id]["stream"]
        codec_type = transcode_meta["codec_type"]
        stream = transcode_meta["sources"]

        if len(stream) == 0:
            raise MissingResourceError(f"Stream {stream_name} is empty")

        if type(stream[0]) is dict:
            part_duration = stream[0]["duration"]["float"]
            parts = [part["source"] for part in stream]
        else:
            dur = stream[0][1]
            if type(dur) is str: dur = int(dur)     ## i don't know if dur is ever not a string, but just anticipate it anyway
            part_duration = dur * float(Fraction(transcode_meta["duration"]["time_base"]))
            parts = [part[0] for part in stream]

        fps = None
        if codec_type == "video":
            fps = self._parse_fps(transcode_meta["rate"])


        return VideoMetadata(
            parts=parts, part_duration=part_duration, fps=fps, codec_type=codec_type
        )

    def _fetch_legacy_vod_metadata(self, q: Content, stream_name: str) -> VideoMetadata:
        """Fetches metadata for legacy VOD content."""
        try:
            streams = q.content_object_metadata(
                metadata_subtree="offerings/default/media_struct/streams",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve streams for {q.qhit}") from e

        assert isinstance(streams, dict)

        if stream_name not in streams:
            raise MissingResourceError(f"Stream {stream_name} not found in {q.qhit}")

        stream = streams[stream_name].get("sources", [])
        if len(stream) == 0:
            raise MissingResourceError(f"Stream {stream_name} is empty")

        parts = [part["source"] for part in stream]

        codec_type = streams[stream_name].get("codec_type", None)
        part_duration = stream[0]["duration"]["float"]

        if codec_type is None:
            raise MissingResourceError(
                f"Codec type not found for stream {stream_name} in {q.qhit}"
            )

        fps = None
        if codec_type == "video":
            fps = self._parse_fps(streams[stream_name]["rate"])

        return VideoMetadata(
            parts=parts, part_duration=part_duration, fps=fps, codec_type=codec_type
        )

    # TODO: may need to have this thing take in a livestream qid instead and then forward to the qwt
    @cache_by_qhash
    def _fetch_livestream_metadata(
        self, q: Content, stream_name: str
    ) -> VideoMetadata:
        """Fetches metadata for livestream content."""
        try:
            periods = q.content_object_metadata(
                metadata_subtree="live_recording/recordings/live_offering",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(
                f"Failed to retrieve periods for live recording {q.qhit}"
            ) from e

        assert isinstance(periods, list)

        if len(periods) == 0:
            raise MissingResourceError(f"Live recording {q.qhit} is empty")

        stream = (
            periods[0]
            .get("sources", {})
            .get(stream_name, {})
            .get("parts", [])
        )
        if len(stream) == 0:
            raise MissingResourceError(
                f"Stream {stream_name} was found in live recording, but no parts were found."
            )

        if stream_name == "video":
            codec_type = "video"
        elif stream_name.startswith("audio"):
            codec_type = "audio"
        else:
            raise BadRequestError(
                f"Invalid stream name for live: {stream_name}. Must be 'video' or start with prefix 'audio'."
            )

        try:
            xc_params = q.content_object_metadata(
                metadata_subtree="live_recording/recording_config/recording_params/xc_params",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(
                f"Failed to retrieve live stream metadata from {q.qhit}"
            ) from e

        fps = None
        if codec_type == "video":
            try:
                live_stream_info = q.content_object_metadata(
                    metadata_subtree="live_recording_config/probe_info/streams",
                    resolve_links=False,
                )
            except HTTPError as e:
                raise HTTPError(
                    f"Failed to retrieve live stream metadata from {q.qhit}"
                ) from e
            
            assert isinstance(live_stream_info, list)

            video_stream_info = None
            for stream_info in live_stream_info:
                if stream_info["codec_type"] == "video":
                    video_stream_info = stream_info
                    break

            if video_stream_info is None:
                raise MissingResourceError(
                    "Video stream not found in live stream metadata"
                )

            fps = self._parse_fps(video_stream_info["frame_rate"])
            assert isinstance(xc_params, dict)
            part_duration = xc_params.get("seg_duration", None)

            if part_duration is None:
                raise MissingResourceError(
                    "Part duration not found in live stream metadata"
                )
            part_duration = float(part_duration)
        else:
            assert isinstance(xc_params, dict)
            sr = xc_params.get("sample_rate", None)
            ts = xc_params.get("audio_seg_duration_ts", None)

            if sr is None or ts is None:
                raise MissingResourceError(
                    "Sample rate or audio segment duration not found in live stream metadata"
                )

            part_duration = int(ts) / int(sr)

        # Filter out parts with finalization_time == 0, meaning the part is still live.
        parts = [
            part["hash"]
            for part in stream
            if part["finalization_time"] != 0 and part["size"] > 0
        ]

        return VideoMetadata(
            parts=parts, part_duration=part_duration, fps=fps, codec_type=codec_type
        )

    @cache_by_qhash
    def _is_live(self, q: Content) -> bool:
        """Check if content is a live stream."""
        if not q.qhit.startswith("tqw__"):
            return False
        try:
            # TODO: make sure works for write token
            q.content_object_metadata(metadata_subtree="live_recording")
        except HTTPError:
            return False
        return True

    @cache_by_qhash
    def _is_legacy_vod(self, q: Content) -> bool:
        """Check if content is legacy VOD format."""
        try:
            q.content_object_metadata(metadata_subtree="transcodes")
        except HTTPError:
            return True
        return False

    def _parse_fps(self, rat: str) -> float:
        """Parse FPS from string format (e.g., '30/1' or '29.97')."""
        if "/" in rat:
            num, den = rat.split("/")
            return float(num) / float(den)
        return float(rat)