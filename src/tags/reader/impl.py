
import re

from src.common.content import Content
from src.tags.reader.abstract import TagReader
from src.tags.datastore.abstract import Datastore
from src.tags.datastore.model import Tag

# matches tracks like "10s" requesting fixed-size intervals of N seconds
INTERVAL_TRACK_PATTERN = re.compile(r"^(\d+)s$")


class TagReaderImpl(TagReader):
    def __init__(
        self,
        q: Content,
        tagstore: Datastore,
        track: str
    ):
        self.q = q
        self.ts = tagstore
        self.track = track

    def read(self) -> list[Tag]:
        return self.ts.find_tags(self.q, track=self.track, limit=100000)


class IntervalTagReader(TagReader):
    """A tag reader that synthesizes tags at fixed time intervals instead of
    querying the tagstore. Selected when the requested track matches a pattern
    like "10s", which yields tags spanning (0, 10000), (10000, 20000), ... up to
    the media duration (all times in milliseconds).
    """

    def __init__(
        self,
        interval_ms: int,
        duration_ms: int
    ):
        assert interval_ms > 0
        self.interval_ms = interval_ms
        self.duration_ms = duration_ms

    def read(self) -> list[Tag]:
        tags: list[Tag] = []
        start = 0
        while start < self.duration_ms:
            end = min(start + self.interval_ms, self.duration_ms)
            tags.append(Tag(
                id=f"interval_{start}_{end}",
                start_time=start,
                end_time=end,
                data="",
                additional_info=None,
                source="interval",
                batch_id="",
            ))
            start += self.interval_ms
        return tags