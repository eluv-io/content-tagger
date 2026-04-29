from typing import Protocol

from src.tags.tagstore.model import Tag

class TagReader(Protocol):
    def read(self) -> list[Tag]: ...