
from dataclasses import dataclass
from src.tagger.fabric_tagging.types import RunConfig

from common_ml.types import Data

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
    
@dataclass
class ImageTagArgs(Data):
    # maps feature name to RunConfig
    features: dict[str, RunConfig]

    # asset file paths to tag relative to the content object e.g. /assets/image.jpg, if empty then we will look in /meta/assets and tag all the image assets located there. 
    assets: list[str] | None

    # replace tag files if they already exist
    replace: bool=False

    @staticmethod
    def from_dict(data: dict) -> 'ImageTagArgs':
        features = {feature: RunConfig(stream='image', **cfg) for feature, cfg in data['features'].items()}
        return ImageTagArgs(features=features, assets=data.get('assets', None), replace=data.get('replace', False))