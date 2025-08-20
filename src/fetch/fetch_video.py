from requests.exceptions import HTTPError
import os
import threading
import tempfile
import shutil
from copy import deepcopy
from common_ml.video_processing import unfrag_video
from loguru import logger

from src.common.content import Content
from src.common.errors import MissingResourceError, BadRequestError
from src.fetch.types import VodDownloadRequest, Source, StreamMetadata
from src.fetch.types import FetcherConfig, DownloadResult
from src.tags.tagstore import FilesystemTagStore


class Fetcher:
    def __init__(
        self,
        config: FetcherConfig,
        tagstore: FilesystemTagStore,
    ):
        self.config = config
        self.tagstore = tagstore
        self.dl_sem = threading.Semaphore(config.max_downloads)

    def download_stream(
        self,
        q: Content,
        req: VodDownloadRequest,
        exit_event: threading.Event | None = None,
    ) -> DownloadResult:
        with self.dl_sem:
            return self._download_stream(q, req, exit_event)

    def _download_stream(
        self,
        q: Content,
        req: VodDownloadRequest,
        exit_event: threading.Event | None = None,
    ) -> DownloadResult:
        req = deepcopy(req)

        if req.start_time is None:
            req.start_time = 0

        if req.end_time is None:
            req.end_time = float("inf")

        stream_metadata = self.fetch_stream_metadata(q, req.stream_name)
        return self._download_parts(q, req, stream_metadata, exit_event)

    def fetch_stream_metadata(self, q: Content, stream_name: str) -> StreamMetadata:
        """Fetches metadata for a stream based on content type."""
        if self._is_live(q):
            return self._fetch_livestream_metadata(q, stream_name)
        elif self._is_legacy_vod(q):
            return self._fetch_legacy_vod_metadata(q, stream_name)
        else:
            return self._fetch_vod_metadata(q, stream_name)

    def _fetch_vod_metadata(self, q: Content, stream_name: str) -> StreamMetadata:
        """Fetches metadata for modern VOD content."""
        try:
            transcodes = q.content_object_metadata(
                metadata_subtree="transcodes", resolve_links=False
            )
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve transcodes for {q.qhit}") from e

        try:
            streams = q.content_object_metadata(
                metadata_subtree="offerings/default/playout/streams",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve streams for {q.qhit}") from e

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
                logger.error(
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

        part_duration = stream[0]["duration"]["float"]

        fps = None
        if codec_type == "video":
            fps = self._parse_fps(transcode_meta["rate"])

        parts = [part["source"] for part in stream]

        return StreamMetadata(
            parts=parts, part_duration=part_duration, fps=fps, codec_type=codec_type
        )

    def _fetch_legacy_vod_metadata(self, q: Content, stream_name: str) -> StreamMetadata:
        """Fetches metadata for legacy VOD content."""
        try:
            streams = q.content_object_metadata(
                metadata_subtree="offerings/default/media_struct/streams",
                resolve_links=False,
            )
        except HTTPError as e:
            raise HTTPError(f"Failed to retrieve streams for {q.qhit}") from e

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

        return StreamMetadata(
            parts=parts, part_duration=part_duration, fps=fps, codec_type=codec_type
        )

    def _fetch_livestream_metadata(
        self, q: Content, stream_name: str
    ) -> StreamMetadata:
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
            part_duration = xc_params.get("seg_duration", None)

            if part_duration is None:
                raise MissingResourceError(
                    "Part duration not found in live stream metadata"
                )
            part_duration = float(part_duration)
        else:
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

        return StreamMetadata(
            parts=parts, part_duration=part_duration, fps=fps, codec_type=codec_type
        )

    def _download_parts(
        self,
        q: Content,
        req: VodDownloadRequest,
        stream_metadata: StreamMetadata,
        exit_event: threading.Event | None = None,
    ) -> DownloadResult:
        """
        Downloads the parts from the stream and returns them. 
        
        If req.replace is True, doesn't return already tagged tags.

        Returns:
            DownloadResult containing successful_sources and failed_part_hashes
        """

        stream_metadata = deepcopy(stream_metadata)

        if not req.replace:
            stream_metadata.parts = self._filter_parts(q, stream_metadata.parts)

        output_path = self.config.parts_path

        tmp_path = tempfile.mkdtemp()
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        successful_sources = []
        failed_parts = []

        for idx, part_hash in enumerate(stream_metadata.parts):
            if exit_event is not None and exit_event.is_set():
                break

            pstart = idx * stream_metadata.part_duration
            pend = (idx + 1) * stream_metadata.part_duration
            idx_str = str(idx).zfill(4)

            # Check if part is within time range
            if not (
                req.start_time <= pstart < req.end_time
            ) and not (req.start_time <= pend < req.end_time):
                continue

            filename = f"{idx_str}_{part_hash}.mp4"
            save_path = os.path.join(output_path, filename)

            # Skip if file exists and not replacing
            if os.path.exists(save_path):
                source = Source(
                    name=filename,
                    filepath=save_path,
                    offset=pstart,
                )
                successful_sources.append(source)
                continue

            logger.info(f"Downloading part {part_hash} for {q.qhit}")
            tmpfile = os.path.join(tmp_path, f"{idx_str}_{part_hash}")

            try:
                q.download_part(save_path=tmpfile, part_hash=part_hash)

                if stream_metadata.codec_type == "video":
                    unfrag_video(tmpfile, save_path)
                else:
                    shutil.move(tmpfile, save_path)

                source = Source(
                    name=filename,
                    filepath=save_path,
                    offset=pstart,
                )
                successful_sources.append(source)

            except Exception as e:
                if os.path.exists(save_path):
                    # Remove the corrupt file if it exists
                    os.remove(save_path)
                failed_parts.append(part_hash)
                logger.error(
                    f"Failed to download part {part_hash} for {q.qhit}: {str(e)}"
                )
                continue

        shutil.rmtree(tmp_path, ignore_errors=True)
        return DownloadResult(
            successful_sources=successful_sources, failed_part_hashes=failed_parts
        )

    def _filter_parts(self, q: Content, parts: list[str]) -> list[str]:
        tagged_parts = self.tagstore.list_tagged_sources(q.qhit)
        return [part for part in parts if part not in tagged_parts]

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