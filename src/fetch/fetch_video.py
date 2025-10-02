from collections import defaultdict
import timeit
from requests.exceptions import HTTPError
import os
import threading
import tempfile
import shutil
from fractions import Fraction
from copy import deepcopy
from common_ml.video_processing import unfrag_video
from common_ml.utils.files import get_file_type, encode_path
from common_ml.utils.metrics import timeit
from loguru import logger

from src.common.content import Content
from src.common.errors import MissingResourceError, BadRequestError
from src.fetch.types import AssetScope, DownloadRequest, VideoScope, DownloadResult, Source, StreamMetadata
from src.fetch.types import FetcherConfig, DownloadResult
from src.tags.tagstore.abstract import Tagstore


class Fetcher:
    def __init__(
        self,
        config: FetcherConfig,
        ts: Tagstore,
    ):
        self.config = config
        self.ts = ts
        # sem to keep total IO reasonable
        self.dl_sem = threading.Semaphore(config.max_downloads)
        # maps (qhit, stream) to lock to prevent unnecessary stream download duplication
        self.stream_locks = defaultdict(threading.Lock)

    def download(
        self,
        q: Content,
        req: DownloadRequest,
        exit_event: threading.Event | None = None,
    ) -> DownloadResult:
        logger.debug(req)
        # TODO: need to flip order of sem and stream locks
        with self.dl_sem:
            stream_key = (q.qhit, req.stream_name)
            with self.stream_locks[stream_key]:
                if req.stream_name == "assets":
                    return self._fetch_assets(q, req)
                return self._download_stream(q, req, exit_event)

    def _download_stream(
        self,
        q: Content,
        req: DownloadRequest,
        exit_event: threading.Event | None = None,
    ) -> DownloadResult:
        req = deepcopy(req)

        scope = req.scope
        assert isinstance(scope, VideoScope)

        stream_metadata = self._fetch_stream_metadata(q, req.stream_name)
        return self._download_parts(q, req, stream_metadata, exit_event)

    def _fetch_stream_metadata(self, q: Content, stream_name: str) -> StreamMetadata:
        """Fetches metadata for a stream based on content type."""
        if self._is_live(q):
            return self._fetch_livestream_metadata(q, stream_name)
        
        if stream_name == "audio":
            # NOTE: kinda janky logic but stream_name "audio" is special-cased to mean "default audio stream"
            logger.info(f"Finding default audio stream for {q.qhit}")
            stream_name = self._find_default_audio_stream(q)

        if self._is_legacy_vod(q):
            return self._fetch_legacy_vod_metadata(q, stream_name)
        else:
            return self._fetch_vod_metadata(q, stream_name)
        
    def _find_default_audio_stream(self, q: Content) -> str:
        streams = q.content_object_metadata(
            metadata_subtree="offerings/default/media_struct/streams",
            resolve_links=False,
        )

        assert isinstance(streams, dict)

        for stream_name, stream_info in streams.items():
            if stream_info.get("codec_type") == "audio" and \
                stream_info.get("language") == "en" and \
                stream_info.get("channels") == 2:
                return stream_name
            
        return "audio"
                
    def _fetch_vod_metadata(self, q: Content, stream_name: str) -> StreamMetadata:
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

        return StreamMetadata(
            parts=parts, part_duration=part_duration, fps=fps, codec_type=codec_type
        )

    def _download_parts(
        self,
        q: Content,
        req: DownloadRequest,
        stream_metadata: StreamMetadata,
        exit_event: threading.Event | None = None,
    ) -> DownloadResult:
        """
        Downloads the parts from the stream and returns them. 
        
        If req.replace is True, doesn't return already tagged tags.

        Returns:
            DownloadResult containing successful_sources and failed_part_hashes
        """

        if stream_metadata.codec_type not in ["video", "audio"]:
            raise BadRequestError(
                f"Invalid codec type for live: {stream_metadata.codec_type}. Must be 'video' or 'audio'."
            )

        output_path = os.path.join(self.config.parts_dir, q.qhit, req.stream_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tmp_path = tempfile.mkdtemp()

        successful_sources = []
        failed_parts = []

        scope = req.scope
        assert isinstance(scope, VideoScope)
        start_time, end_time = scope.start_time, scope.end_time

        to_download = stream_metadata.parts

        logger.info(f"Downloading stream {req.stream_name} with {len(to_download)} total parts.")

        tagged_parts = []
        if req.preserve_track:
            existing_tags = self.ts.find_tags(
                author=self.config.author, 
                qhit=q.qid, 
                stream=req.stream_name, 
                track=req.preserve_track,
                auth=q._client.token
            )
            tagged_parts = {tag.source for tag in existing_tags}

        if req.preserve_track and tagged_parts:
            logger.info(f"Filtering {len(tagged_parts)} already tagged parts")

        for idx, part_hash in enumerate(to_download):
            if exit_event is not None and exit_event.is_set():
                break

            if part_hash in tagged_parts:
                continue

            pstart = idx * stream_metadata.part_duration
            pend = (idx + 1) * stream_metadata.part_duration
            idx_str = str(idx).zfill(4)

            # Check if part is within time range
            if not (
                start_time <= pstart < end_time
            ) and not (start_time <= pend < end_time):
                continue

            filename = f"{idx_str}_{part_hash}{'.mp4' if stream_metadata.codec_type == 'video' else '.m4a'}"
            save_path = os.path.join(output_path, filename)

            # Skip if file exists and not replacing
            if os.path.exists(save_path):
                source = Source(
                    name=part_hash,
                    filepath=save_path,
                    offset=pstart,
                )
                successful_sources.append(source)
                continue

            tmpfile = os.path.join(tmp_path, f"{idx_str}_{part_hash}")

            try:
                q.download_part(save_path=tmpfile, part_hash=part_hash)

                if stream_metadata.codec_type == "video":
                    unfrag_video(tmpfile, save_path)
                else:
                    shutil.move(tmpfile, save_path)

                source = Source(
                    name=part_hash,
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

            # check that length of the file is equal to the part length
            # if not last_part and stream_metadata.codec_type == "video":
            #     actual_duration = get_video_length(save_path)
            #     assert abs(actual_duration - stream_metadata.part_duration) < 1e-3

            # TODO: check for audio as well. 

        shutil.rmtree(tmp_path, ignore_errors=True)

        return DownloadResult(
            successful_sources=successful_sources, failed=failed_parts, stream_meta=stream_metadata
        )

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
    
    def _fetch_assets(
        self, 
        q: Content, 
        req: DownloadRequest
    ) -> DownloadResult:
        output_path = os.path.join(self.config.parts_dir, q.qhit, "assets")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        scope = req.scope
        assert isinstance(scope, AssetScope)

        if scope.assets is None:
            assets_meta = q.content_object_metadata(metadata_subtree='assets')
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

        total_assets = len(assets)
        assets = [asset for asset in assets if get_file_type(asset) == "image"]
        logger.info(f"Found {len(assets)} image assets out of {total_assets} assets for {q.qhit}")
        if len(assets) == 0:
            raise MissingResourceError(f"No image assets found in {q.qhit}")

        tagged_assets = []
        if req.preserve_track:
            existing_tags = self.ts.find_tags(author=self.config.author, qhit=q.qhit, stream="assets", track=req.preserve_track, auth=q._client.token)
            tagged_assets = [tag.source for tag in existing_tags]

        if tagged_assets:
            logger.info(f"Filtering {len(tagged_assets)} already tagged assets")

        # Filter out already tagged assets
        assets_to_process = [asset for asset in assets if asset not in tagged_assets]
        
        # Separate assets that already exist vs need downloading
        already_downloaded = []
        to_download = []
        
        for asset in assets_to_process:
            save_path = os.path.join(output_path, encode_path(asset))
            if os.path.exists(save_path):
                already_downloaded.append(asset)
            else:
                to_download.append((asset, save_path))

        if already_downloaded:
            logger.info(f"{len(already_downloaded)} assets already retrieved for {q.qhit}")

        logger.info(f"{len(to_download)} assets need to be downloaded for {q.qhit}")

        # Download new assets
        assets_to_download = [asset for asset, _ in to_download]
        newly_downloaded = self._download_concurrent(q, to_download, output_path)

        # Combine all successful assets (already downloaded + newly downloaded)
        all_successful_assets = already_downloaded + newly_downloaded
        
        # Create successful sources for all available assets
        successful_sources = [
            Source(name=asset, filepath=os.path.join(output_path, encode_path(asset)), offset=0) 
            for asset in all_successful_assets
        ]
        
        # Failed assets are those we tried to download but couldn't
        failed_assets = set(assets_to_download) - set(newly_downloaded)
        failed = list(failed_assets)
        
        return DownloadResult(successful_sources=successful_sources, failed=failed, stream_meta=None)

    def _download_concurrent(self, q: Content, file_jobs: list[tuple[str, str]], output_path: str) -> list[str]:
        with timeit("Downloading assets"):
            status = q.download_files(dest_path=output_path, file_jobs=file_jobs)
        assert isinstance(status, list) and len(status) == len(file_jobs)
        return [asset for (asset, _), error in zip(file_jobs, status) if error is None]