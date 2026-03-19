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
        q: Content,
        scope: TimeRangeScope,
        meta: VideoMetadata,
        ignore_sources: list[str],
        output_dir: str,
        exit: threading.Event | None = None
    ):
        self.q = q
        self.scope = scope
        self.meta = meta
        self.output_dir = output_dir
        self.ignore_sources = set(ignore_sources)
        self.exit = exit

    def metadata(self) -> MediaMetadata:
        return deepcopy(self.meta)
    
    def download(self) -> DownloadResult:
        content_duration = self.meta.part_duration * len(self.meta.parts)

        logger.debug(f"Live stream duration based on metadata: {content_duration} seconds")
        logger.debug(f"Requested time range: {self.scope.start_time} - {self.scope.end_time} seconds")
        
        start_time = self.scope.start_time or 0
        end_time = min(self.scope.end_time or math.ceil(content_duration), math.ceil(content_duration))

        # clamp to chunk_size boundaries, but don't go beyond content duration
        start_time = (start_time // self.scope.chunk_size) * self.scope.chunk_size
        end_time = math.ceil(end_time / self.scope.chunk_size) * self.scope.chunk_size

        logger.debug(f"Padded time range: {start_time} - {end_time} seconds")

        sources = []
        t_iterator = start_time
        while t_iterator < end_time:
            t = t_iterator
            t_iterator += self.scope.chunk_size
            this_start = t * 1000
            this_end = (t + self.scope.chunk_size) * 1000

            intv = f'{"%010d" % this_start}_{"%010d" % this_end}'
            output_path = os.path.join(self.output_dir, f'{intv}.json')

            if intv not in self.ignore_sources:
                sources.append(Source(
                    name=intv,
                    filepath=output_path,
                    offset=0,
                    wall_clock=None
                ))

                with open(output_path, 'w') as f:
                    content = {
                        "iq": self.q.qid,
                        "token": self.q.token,
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