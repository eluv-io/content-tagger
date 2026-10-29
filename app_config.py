from dataclasses import dataclass, field
from typing import Literal
import yaml
import os
from dacite import from_dict

from src.common.content import ContentConfig
from src.common.logging import LoggingConfig
from src.common.model import ModelConfig
from src.fetch.model import FetcherConfig
from src.tag_containers.model import RegistryConfig
from src.tagging.scheduling.model import SysConfig
from src.tagging.tag_runner import TagRunnerConfig
from src.tagging.fabric_tagging.queue.model import JobStoreConfig
from src.tags.tagstore.model import TagstoreConfig
from src.tags.vectorstore.model import VectorstoreConfig
from src.tagging.fabric_tagging.model import TaggerWorkerConfig
from src.tags.track_resolver import LabelResolverConfig
from src.status.get_info import UserInfoResolverConfig

@dataclass
class AppConfig:
    root_dir: str
    content: ContentConfig
    jobstore: JobStoreConfig
    tagstore: TagstoreConfig
    system: SysConfig
    fetcher: FetcherConfig
    container_registry: RegistryConfig
    model_configs: dict[str, ModelConfig]
    tagger: TaggerWorkerConfig
    label_resolver: LabelResolverConfig
    tag_runner: TagRunnerConfig
    user_info_resolver: UserInfoResolverConfig
    vectorstore: VectorstoreConfig = field(default_factory=VectorstoreConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def from_yaml(filename: str) -> 'AppConfig':
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        if "root_dir" not in data:
            data["root_dir"] = os.getcwd()
        data = AppConfig._resolve_paths(data, data["root_dir"])
        cfg = from_dict(AppConfig, data)
        cfg._inject_container_env()
        return cfg

    def _inject_container_env(self) -> None:
        """Default service URLs passed to containers; explicit container_registry.env wins"""
        defaults = {
            "ELV_TAGSTORE_URL": self.tagstore.base_url,
            "ELV_VECTORSTORE_URL": self.vectorstore.base_url,
        }
        env = {**defaults, **self.container_registry.env}
        self.container_registry.env = {k: v for k, v in env.items() if v}
    
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