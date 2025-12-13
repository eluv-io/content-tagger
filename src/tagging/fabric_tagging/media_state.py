from dataclasses import dataclass

from src.fetch.model import FetchSession, Source
    
@dataclass
class MediaState:
    # used to keep track of sources that need to be downloaded for status
    # also used to help the uploader know how to compute the offset
    downloaded: list[Source]
    # used to get media
    worker: FetchSession