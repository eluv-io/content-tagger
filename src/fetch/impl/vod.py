

import os
import shutil
from loguru import logger
import tempfile
from copy import deepcopy
import threading

from common_ml.video_processing import unfrag_video

from src.common.content import QAPI
from src.fetch.model import *
from src.fetch.rate_limit import FetchRateLimiter
from src.common.errors import BadRequestError

logger = logger.bind(module="fetch vod")


class VodWorker(FetchSession):
    def __init__(
        self,
        qapi: QAPI,
        scope: VideoScope,
        rate_limiter: FetchRateLimiter,
        meta: VideoMetadata,
        ignore_sources: list[str],
        output_dir: str,
        exit: threading.Event | None = None,
        batch_size: int = 25
    ):
        self.qapi = qapi
        self.scope = scope
        self.rl = rate_limiter
        self.codec_type = meta.codec_type
        self.part_duration = meta.part_duration
        self.output_dir = output_dir
        self.ignore_sources = set(ignore_sources)
        self.exit = exit
        self._fps = meta.fps
        self.batch_size = batch_size

        start_time, end_time = scope.start_time, scope.end_time
        # Pre-compute (original_idx, part_hash) for parts that pass all filters.
        # original_idx is preserved so _download can compute correct offsets.
        self._filtered_parts: list[tuple[int, str]] = [
            (idx, part_hash)
            for idx, part_hash in enumerate(meta.parts)
            if part_hash not in self.ignore_sources
            and (
                start_time <= idx * self.part_duration < end_time
                or start_time <= (idx + 1) * self.part_duration < end_time
            )
        ]
        self._cursor = 0

    def metadata(self) -> MediaMetadata:
        return MediaMetadata(
            sources=[part_hash for _, part_hash in self._filtered_parts],
            fps=self._fps
        )

    def download(self) -> DownloadResult:
        with self.rl.permit((self.qapi.id(), str(self.scope.stream))):
            return self._download()

    @property
    def path(self) -> str:
        return self.output_dir

    def _download(self) -> DownloadResult:
        """
        Downloads the parts from the stream and returns them. 
        
        If req.replace is True, doesn't return already tagged tags.

        Returns:
            DownloadResult containing successful_sources and failed_part_hashes
        """

        if self.codec_type not in ["video", "audio"]:
            raise BadRequestError(
                f"Invalid codec type for live: {self.codec_type}. Must be 'video' or 'audio'."
            )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        tmp_path = tempfile.mkdtemp()

        successful_sources = []
        failed_parts = []

        batch = self._filtered_parts[self._cursor:self._cursor + self.batch_size]

        logger.info(
            f"Downloading stream {self.scope.stream}: "
            f"batch {self._cursor // self.batch_size + 1}, {len(batch)} parts "
            f"(of {len(self._filtered_parts)} filtered)."
        )

        for idx, part_hash in batch:
            if self.exit and self.exit.is_set():
                break

            pstart = idx * self.part_duration
            pend = (idx + 1) * self.part_duration
            idx_str = str(idx).zfill(4)

            filename = f"{idx_str}_{part_hash}{'.mp4' if self.codec_type == 'video' else '.m4a'}"
            save_path = os.path.join(self.output_dir, filename)

            if os.path.exists(save_path):
                source = Source(
                    name=part_hash,
                    filepath=save_path,
                    offset=int(pstart * 1000),
                    wall_clock=None
                )
                successful_sources.append(source)
                continue

            tmpfile = os.path.join(tmp_path, f"{idx_str}_{part_hash}")

            try:
                self.qapi.download_part(save_path=tmpfile, part_hash=part_hash)

                if self.codec_type == "video":
                    unfrag_video(tmpfile, save_path)
                else:
                    shutil.move(tmpfile, save_path)

                source = Source(
                    name=part_hash,
                    filepath=save_path,
                    offset=int(pstart * 1000),
                    wall_clock=None
                )
                successful_sources.append(source)

            except Exception as e:
                if os.path.exists(save_path):
                    # Remove the corrupt file if it exists
                    os.remove(save_path)
                failed_parts.append(part_hash)
                logger.error(
                    f"Failed to download part {part_hash} for {self.qapi.id()}: {str(e)}"
                )
                continue

            # check that length of the file is equal to the part length
            # if not last_part and self.codec_type == "video":
            #     actual_duration = get_video_length(save_path)
            #     assert abs(actual_duration - self.part_duration) < 1e-3

            # TODO: check for audio as well. 

        self._cursor += len(batch)
        shutil.rmtree(tmp_path, ignore_errors=True)

        return DownloadResult(
            sources=successful_sources,
            failed=failed_parts,
            done=self._cursor >= len(self._filtered_parts)
        )