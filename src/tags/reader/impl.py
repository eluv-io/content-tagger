
from src.common.content import Content
from src.tags.reader.abstract import TagReader
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import Tag


class TagReaderImpl(TagReader):
    def __init__(
        self,
        q: Content,
        tagstore: Tagstore,
        track: str
    ):
        self.q = q
        self.ts = tagstore
        self.track = track
    
    def read(self) -> list[Tag]:
        return self.ts.find_tags(self.q, track=self.track, limit=100000)