from src.common.content import Content
from src.tags.reader.abstract import TagReader
from src.tags.reader.impl import TagReaderImpl
from src.tags.datastore.abstract import Datastore


class TagReaderFactory:
    def __init__(self, tagstore: Datastore):
        self.ts = tagstore

    def get(self, q: Content, track: str) -> TagReader:
        return TagReaderImpl(q, self.ts, track)