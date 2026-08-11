
from src.tags.datastore.abstract import Datastore
from src.tags.tagstore.model import TagstoreConfig
from src.tags.tagstore.filesystem_tagstore import FilesystemTagStore
from src.tags.tagstore.rest_tagstore import RestTagstore

def create_tagstore(cfg: TagstoreConfig) -> Datastore:
    if cfg.base_url:
        return RestTagstore(cfg.base_url, cfg.timeout)
    else:
        return FilesystemTagStore(cfg.base_dir)
