from typing import Protocol

from src.tags.datastore.model import Tag

class TagReader(Protocol):
    def read(self) -> list[Tag]: ...