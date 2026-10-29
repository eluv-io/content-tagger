
from dataclasses import dataclass

## Config

@dataclass
class VectorstoreConfig:
    base_url: str=""
    timeout: int=10
