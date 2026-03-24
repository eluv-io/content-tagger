from dataclasses import dataclass
import yaml
import os
from dacite import from_dict

from src.common.content import ContentConfig
from src.fetch.model import FetcherConfig
from src.tag_containers.model import RegistryConfig
from src.tagging.scheduling.model import SysConfig
from src.tagging.tag_runner import TagRunnerConfig
from src.tagging.fabric_tagging.queue.model import JobStoreConfig
from src.tags.tagstore.model import TagstoreConfig
from src.tagging.fabric_tagging.model import TaggerWorkerConfig
from src.tags.track_resolver import TrackResolverConfig

@dataclass
class AppConfig:
    root_dir: str
    content: ContentConfig
    jobstore: JobStoreConfig
    tagstore: TagstoreConfig
    system: SysConfig
    fetcher: FetcherConfig
    container_registry: RegistryConfig
    tagger: TaggerWorkerConfig
    track_resolver: TrackResolverConfig
    tag_runner: TagRunnerConfig

    @staticmethod
    def from_yaml(filename: str) -> 'AppConfig':
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        if "root_dir" not in data:
            data["root_dir"] = os.getcwd()
        data = AppConfig._resolve_paths(data, data["root_dir"])
        return from_dict(AppConfig, data)
    
    @staticmethod
    def _resolve_paths(data: dict, root: str) -> dict:

        def resolve_path(value: str) -> str:
            if value.startswith('/'):
                return value
            return f"{root}/{value}"

        def resolve_config(config: dict) -> dict:
            for key, value in config.items():
                if isinstance(value, str) and (key.endswith('_dir') or key.endswith('_path')):
                    config[key] = resolve_path(value)
                elif isinstance(value, dict):
                    config[key] = resolve_config(value)
            return config

        return resolve_config(data)