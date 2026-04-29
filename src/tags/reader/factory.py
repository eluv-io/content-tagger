from src.common.content import Content
from src.tags.reader.abstract import TagReader
from src.tags.reader.impl import TagReaderImpl
from src.tags.tagstore.abstract import Tagstore


class TagReaderFactory:
    def __init__(self, tagstore: Tagstore):
        self.ts = tagstore

    def get(self, q: Content, track: str) -> TagReader:
        return TagReaderImpl(q, self.ts, track)