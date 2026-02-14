

import os
import shutil
from loguru import logger
import tempfile
from copy import deepcopy
import threading

from common_ml.video_processing import unfrag_video

from src.common.content import Content
from src.fetch.model import *
from src.fetch.rate_limit import FetchRateLimiter
from src.common.errors import BadRequestError

logger = logger.bind(module="fetch vod")


class VodWorker(FetchSession):
    def __init__(
        self,
        q: Content,
        scope: VideoScope,
        rate_limiter: FetchRateLimiter,
        meta: VideoMetadata, 
        ignore_sources: list[str],
        output_dir: str,
        exit: threading.Event | None = None
    ):
        self.q = q
        self.scope = scope
        self.rl = rate_limiter
        self.meta = meta
        self.output_dir = output_dir
        self.ignore_sources = set(ignore_sources)
        self.exit = exit

    def metadata(self) -> VideoMetadata:
        return deepcopy(self.meta)

    def download(self) -> DownloadResult:
        with self.rl.permit((self.q.qhit, str(self.scope.stream))):
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

        if self.meta.codec_type not in ["video", "audio"]:
            raise BadRequestError(
                f"Invalid codec type for live: {self.meta.codec_type}. Must be 'video' or 'audio'."
            )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        tmp_path = tempfile.mkdtemp()

        successful_sources = []
        failed_parts = []

        scope = self.scope
        start_time, end_time = scope.start_time, scope.end_time

        to_download = self.meta.parts

        logger.info(f"Downloading stream {self.scope.stream} with {len(to_download)} total parts.")

        if self.ignore_sources:
            logger.info(f"Filtering {len(self.ignore_sources)} already tagged sources")

        for idx, part_hash in enumerate(to_download):
            if self.exit and self.exit.is_set():
                break

            if part_hash in self.ignore_sources:
                continue

            pstart = idx * self.meta.part_duration
            pend = (idx + 1) * self.meta.part_duration
            idx_str = str(idx).zfill(4)

            # Check if part is within time range
            if not (
                start_time <= pstart < end_time
            ) and not (start_time <= pend < end_time):
                continue

            filename = f"{idx_str}_{part_hash}{'.mp4' if self.meta.codec_type == 'video' else '.m4a'}"
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
                self.q.download_part(save_path=tmpfile, part_hash=part_hash)

                if self.meta.codec_type == "video":
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
                    f"Failed to download part {part_hash} for {self.q.qhit}: {str(e)}"
                )
                continue

            # check that length of the file is equal to the part length
            # if not last_part and self.meta.codec_type == "video":
            #     actual_duration = get_video_length(save_path)
            #     assert abs(actual_duration - self.meta.part_duration) < 1e-3

            # TODO: check for audio as well. 

        shutil.rmtree(tmp_path, ignore_errors=True)

        return DownloadResult(
            sources=successful_sources, 
            failed=failed_parts,
            done=True
        )