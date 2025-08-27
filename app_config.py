from dataclasses import dataclass
import yaml
from dacite import from_dict

from src.common.content import ContentConfig
from src.fetch.types import FetcherConfig
from src.tag_containers.types import RegistryConfig
from src.tagger.system_tagging.types import SysConfig
from src.tags.tagstore.types import TagStoreConfig

@dataclass
class AppConfig:
    content: ContentConfig
    tagstore: TagStoreConfig
    system: SysConfig
    fetcher: FetcherConfig
    container_registry: RegistryConfig

    @staticmethod
    def from_yaml(filename: str) -> 'AppConfig':
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        return from_dict(AppConfig, data)