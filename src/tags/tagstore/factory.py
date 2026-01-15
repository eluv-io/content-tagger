
from src.tags.tagstore.abstract import Tagstore
from src.tags.tagstore.model import TagstoreConfig
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.tags.tagstore.rest_tagstore import RestTagstore

def create_tagstore(cfg: TagstoreConfig) -> Tagstore:
    if cfg.base_url:
        return RestTagstore(cfg.base_url, cfg.timeout)
    else:
        return FilesystemTagStore(cfg.base_dir)