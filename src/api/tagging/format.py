
from dataclasses import dataclass
from common_ml.types import Data
from src.fetch.model import AssetScope
from src.tagging.fabric_tagging.model import TagArgs

@dataclass
class ModelParams:
    stream: str
    run_config: dict

@dataclass
class TagAPIArgs(Data):
    # maps feature name to RunConfig
    features: dict[str, ModelParams]
    # start_time in milliseconds (defaults to 0)
    start_time: int | None=None
    # end_time in milliseconds (defaults to entire content)
    end_time: int | None=None
    # replace tag files if they already exist
    replace: bool=False

    @staticmethod
    def from_dict(data: dict) -> 'TagAPIArgs':
        features = {feature: ModelParams(**cfg) for feature, cfg in data['features'].items()}
        return TagAPIArgs(features=features, start_time=data.get('start_time', None), end_time=data.get('end_time', None), replace=data.get('replace', False))

    def __str__(self) -> str:
        return f"TagAPIArgs(features={self.features}, start_time={self.start_time}, end_time={self.end_time}, replace={self.replace})"


@dataclass
class ImageTagAPIArgs(Data):
    # maps feature name to ModelParams
    features: dict[str, ModelParams]

    # asset file paths to tag relative to the content object e.g. /assets/image.jpg, if empty then we will look in /meta/assets and tag all the image assets located there. 
    assets: list[str] | None

    # replace tag files if they already exist
    replace: bool=False

    @staticmethod
    def from_dict(data: dict) -> 'ImageTagAPIArgs':
        features = {feature: ModelParams(stream='assets', **cfg) for feature, cfg in data['features'].items()}
        return ImageTagAPIArgs(features=features, assets=data.get('assets', None), replace=data.get('replace', False))