
from dataclasses import dataclass

## Config

@dataclass
class TagstoreConfig:
    base_dir: str=""
    base_url: str=""
    timeout: int=30
    auth_token: str | None = None