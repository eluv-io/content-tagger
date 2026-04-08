
from requests.exceptions import HTTPError
from fractions import Fraction
import threading

from src.common.logging import logger

from src.common.content import QAPI, Content, QAPIFactory
from src.common.errors import MissingResourceError, BadRequestError
from src.fetch.impl.assets import AssetWorker
from src.fetch.impl.live import LiveWorker
from src.fetch.impl.processors import SkipWorker
from src.fetch.impl.tag_aligned import TagAlignedFetcher
from src.fetch.impl.vod import VodWorker
from src.fetch.model import *
from src.fetch.cache import cache_by_qhash
from src.fetch.rate_limit import FetchRateLimiter
from src.tags.reader.impl import TagReaderImpl
from src.tags.tagstore.abstract import Tagstore

class FetchFactory:
    def __init__(
        self,
        config: FetcherConfig,
        # we need this dependency for the tag-aligned fetcher
        ts: Tagstore,
        qfactory: QAPIFactory
    ):
        self.config = config
        self.ts = ts
        self.qfactory = qfactory
        # help rate limit & avoid double downloads
        self.rl = FetchRateLimiter(config.max_downloads)

    def get_session(
        self, 
        q: Content,
        req: DownloadRequest, 
        exit: threading.Event | None = None
    ) -> FetchSession:

        qapi = self.qfactory.create(q)

        if isinstance(req.scope, VideoScope):
            meta = self._fetch_stream_metadata(qapi, req.scope.stream)
            return VodWorker(
                qapi=qapi,
                scope=req.scope,
                rate_limiter=self.rl,
                meta=meta,
                ignore_sources=req.ignore_sources,
                output_dir=req.output_dir,
                exit=exit
            )
        elif isinstance(req.scope, AssetScope):
            meta = self._fetch_asset_metadata(qapi, req.scope)
            return AssetWorker(
                qapi=qapi,
                scope=req.scope,
                rate_limiter=self.rl,
                meta=meta,
                ignore_assets=req.ignore_sources,
                output_dir=req.output_dir,
                exit=exit
            )
        elif isinstance(req.scope, LiveScope):
            # TODO: fix fps
            meta = MediaMetadata(sources=[], fps=50)
            return LiveWorker(
                qapi=qapi,
                scope=req.scope,
                rate_limiter=self.rl,
                meta=meta,
                output_dir=req.output_dir,
                ignore_sources=req.ignore_sources,
                exit=exit
            )
        elif isinstance(req.scope, TimeRangeScope):
            meta = self._fetch_stream_metadata(qapi, req.scope.stream)
            return SkipWorker(
                scope=req.scope,
                meta=meta,
                ignore_sources=req.ignore_sources,
                output_dir=req.output_dir,
                exit=exit
            )
        elif isinstance(req.scope, TagAlignedScope):
            meta = self._fetch_stream_metadata(qapi, req.scope.stream)
            # create a normal Vodworker which is called to generate the tag-aligned media
            video_scope = VideoScope(
                stream=req.scope.stream,
                start_time=req.scope.start_time,
                end_time=req.scope.end_time,
            )
            vod_worker = VodWorker(
                qapi=qapi,
                scope=video_scope,
                rate_limiter=self.rl,
                meta=meta,
                ignore_sources=req.ignore_sources,
                output_dir=req.output_dir,
                exit=exit
            )
            tr = TagReaderImpl(
                q=q,
                tagstore=self.ts,
                track=req.scope.track
            )
            return TagAlignedFetcher(
                tr=tr,
                vod=vod_worker
            )
        else:
            raise BadRequestError(f"Unknown scope type: {type(req.scope)}")

    
    def _fetch_asset_metadata(self, qapi: QAPI, scope: AssetScope) -> MediaMetadata:
        # Get list of assets to download
        if scope.assets is None:
            assets_meta = qapi.content_object_metadata(metadata_subtree='assets')
            assert isinstance(assets_meta, dict)
            assets_meta = list(assets_meta.values())
            assets = []
            for ameta in assets_meta:
                filepath = ameta.get("file")["/"]
                assert filepath.startswith("./files/")
                # strip leading term
                filepath = filepath[8:]
                assets.append(filepath)
        else:
            assets = scope.assets
        return MediaMetadata(sources=assets, fps=None)

    @cache_by_qhash
    def _fetch_stream_metadata(self, qapi: QAPI, stream_name: str) -> VideoMetadata:
        """Fetches metadata for a stream based on content type."""
        if self._is_live(qapi):
            return self._fetch_livestream_metadata(qapi, stream_name)

        if self._is_legacy_vod(qapi):
            return self._fetch_legacy_vod_metadata(qapi, stream_name)
        else:
            try:
                return self._fetch_vod_metadata(qapi, stream_name)
            except MissingResourceError as e:
                # sometimes vod metadata is a mix of legacy and modern format - this is a quick fix
                logger.error(f"fetching modern format vod metadata failed, trying legacy format", error=str(e))
                return self._fetch_legacy_vod_metadata(qapi, stream_name)

    def _fetch_vod_metadata(self, qapi: QAPI, stream_name: str) -> VideoMetadata:
        """Fetches metadata for modern VOD content."""
        transcodes = qapi.content_object_metadata(
            metadata_subtree="transcodes", resolve_links=True
        )

        assert isinstance(transcodes, dict)

        streams = qapi.content_object_metadata(
            metadata_subtree="offerings/default/playout/streams",
            resolve_links=True,
        )
        
        assert isinstance(streams, dict)

        if stream_name not in streams:
            raise MissingResourceError(f"Stream {stream_name} not found in {qapi.id()}")

        representations = streams[stream_name].get("representations", {})

        transcode_id = None
        for repr in representations.values():
            tid = repr["transcode_id"]
            if transcode_id is None:
                transcode_id = tid
                continue
            if tid != transcode_id:
                logger.warning(
                    f"Multiple transcode_ids found for stream {stream_name} in {qapi.id()}! Continuing with the first one found."
                )

        if transcode_id is None:
            raise MissingResourceError(
                f"Transcode_id not found for stream {stream_name} in {qapi.id()}"
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

    def _fetch_legacy_vod_metadata(self, qapi: QAPI, stream_name: str) -> VideoMetadata:
        """Fetches metadata for legacy VOD content."""

        streams = qapi.content_object_metadata(
            metadata_subtree="offerings/default/media_struct/streams",
            resolve_links=True,
        )

        assert isinstance(streams, dict)

        if stream_name not in streams:
            raise MissingResourceError(f"Stream {stream_name} not found in {qapi.id()}")

        stream = streams[stream_name].get("sources", [])
        if len(stream) == 0:
            raise MissingResourceError(f"Stream {stream_name} is empty")

        parts = [part["source"] for part in stream]

        codec_type = streams[stream_name].get("codec_type", None)
        part_duration = stream[0]["duration"]["float"]

        if codec_type is None:
            raise MissingResourceError(
                f"Codec type not found for stream {stream_name} in {qapi.id()}"
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
        self, qapi: QAPI, stream_name: str
    ) -> VideoMetadata:
        """Fetches metadata for livestream content."""
        try:
            periods = qapi.content_object_metadata(
                metadata_subtree="live_recording/recordings/live_offering",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(
                f"Failed to retrieve periods for live recording {qapi.id()}"
            ) from e

        assert isinstance(periods, list)

        if len(periods) == 0:
            raise MissingResourceError(f"Live recording {qapi.id()} is empty")

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
            xc_params = qapi.content_object_metadata(
                metadata_subtree="live_recording/recording_config/recording_params/xc_params",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(
                f"Failed to retrieve live stream metadata from {qapi.id()}"
            ) from e

        fps = None
        if codec_type == "video":
            try:
                live_stream_info = qapi.content_object_metadata(
                    metadata_subtree="live_recording_config/probe_info/streams",
                    resolve_links=False,
                )
            except HTTPError as e:
                raise HTTPError(
                    f"Failed to retrieve live stream metadata from {qapi.id()}"
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
    def _is_live(self, qapi: QAPI) -> bool:
        """Check if content is a live stream."""
        if not qapi.id().startswith("tqw__"):
            return False
        try:
            # TODO: make sure works for write token
            qapi.content_object_metadata(metadata_subtree="live_recording")
        except HTTPError:
            return False
        return True

    @cache_by_qhash
    def _is_legacy_vod(self, qapi: QAPI) -> bool:
        """Check if content is legacy VOD format."""
        try:
            qapi.content_object_metadata(metadata_subtree="transcodes")
        except HTTPError:
            return True
        return False

    def _parse_fps(self, rat: str) -> float:
        """Parse FPS from string format (e.g., '30/1' or '29.97')."""
        if "/" in rat:
            num, den = rat.split("/")
            return float(num) / float(den)
        return float(rat)