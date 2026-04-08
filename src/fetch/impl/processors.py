from copy import deepcopy
import json
import math
import os
import threading
from loguru import logger

from src.common.content import Content
from src.fetch.model import DownloadResult, FetchSession, MediaMetadata, Source, TimeRangeScope, VideoMetadata


class SkipWorker(FetchSession):
    """
    Simple fetcher than simply returns a set of intervals as sources and let's the model handle fetching.
    """
    def __init__(
        self,
        scope: TimeRangeScope,
        meta: VideoMetadata,
        ignore_sources: list[str],
        output_dir: str,
        exit: threading.Event | None = None
    ):
        self.scope = scope
        self.meta = meta
        self.part_duration = meta.part_duration
        self.output_dir = output_dir
        self.ignore_sources = set(ignore_sources)
        self.exit = exit

    def _intervals(self) -> list[tuple[int, int, str]]:
        """Return (start_ms, end_ms, name) for each chunk in the time range."""
        content_duration = self.part_duration * len(self.meta.parts)
        start_time = self.scope.start_time or 0
        end_time = min(self.scope.end_time or math.ceil(content_duration), math.ceil(content_duration))

        # clamp to chunk_size boundaries
        start_time = (start_time // self.scope.chunk_size) * self.scope.chunk_size
        end_time = math.ceil(end_time / self.scope.chunk_size) * self.scope.chunk_size

        intervals = []
        t = start_time
        while t < end_time:
            this_start = int(t * 1000)
            this_end = int((t + self.scope.chunk_size) * 1000)
            name = f'{"%010d" % this_start}_{"%010d" % this_end}'
            if name not in self.ignore_sources:
                intervals.append((this_start, this_end, name))
            t += self.scope.chunk_size
        return intervals

    def metadata(self) -> MediaMetadata:
        return MediaMetadata(
            sources=[name for _, _, name in self._intervals()],
            fps=None
        )
    
    def download(self) -> DownloadResult:
        logger.debug(f"Live stream duration based on metadata: {self.part_duration * len(self.meta.parts)} seconds")
        logger.debug(f"Requested time range: {self.scope.start_time} - {self.scope.end_time} seconds")

        sources = []
        for this_start, this_end, intv in self._intervals():
            output_path = os.path.join(self.output_dir, f'{intv}.json')

            sources.append(Source(
                name=intv,
                filepath=output_path,
                offset=0,
                wall_clock=None
            ))

            with open(output_path, 'w') as f:
                content = {
                    "start_time": this_start,
                    "end_time": this_end
                }
                json.dump(content, f)
        
        
        return DownloadResult(
            sources=sources,
            failed=[],
            done=True
        )
    
    @property
    def path(self) -> str:
        return self.output_dir