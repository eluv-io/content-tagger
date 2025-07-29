
from dataclasses import dataclass
from src.tagger.jobs import RunConfig

from common_ml.types import Data

# TagArgs represents the request body for the /tag endpoint
@dataclass
class TagArgs(Data):
    # maps feature name to RunConfig
    features: dict[str, RunConfig]
    # start_time in milliseconds (defaults to 0)
    start_time: int | None=None
    # end_time in milliseconds (defaults to entire content)
    end_time: int | None=None
    # replace tag files if they already exist
    replace: bool=False

    @staticmethod
    def from_dict(data: dict) -> 'TagArgs':
        features = {feature: RunConfig(**cfg) for feature, cfg in data['features'].items()}
        return TagArgs(features=features, start_time=data.get('start_time', None), end_time=data.get('end_time', None), replace=data.get('replace', False))