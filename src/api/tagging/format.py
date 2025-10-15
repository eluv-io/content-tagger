
import dacite

from dataclasses import dataclass, field
from common_ml.types import Data

@dataclass
class ModelParams(Data):
    stream: str | None = None
    model: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict) -> 'ModelParams':
        return ModelParams(stream=data.get('stream'), model=data.get('model', {}))

@dataclass
class TagAPIArgs(Data):
    # maps feature name to ModelParams
    features: dict[str, ModelParams]
    # start_time in milliseconds (defaults to 0)
    start_time: int | None=None
    # end_time in milliseconds (defaults to entire content)
    end_time: int | None=None
    # replace tag files if they already exist
    replace: bool=False

    @staticmethod
    def from_dict(data: dict) -> 'TagAPIArgs':
        features = {feature: ModelParams.from_dict(cfg) for feature, cfg in data['features'].items()}
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
        features = {feature: ModelParams.from_dict(cfg) for feature, cfg in data['features'].items()}
        return ImageTagAPIArgs(features=features, assets=data.get('assets', None), replace=data.get('replace', False))
    
@dataclass
class LiveTagAPIArgs(Data):
    # maps feature name to ModelParams
    features: dict[str, ModelParams]

    segment_length: int = 4
    max_duration: int | None = None

    @staticmethod
    def from_dict(data: dict) -> 'LiveTagAPIArgs':
        return dacite.from_dict(data_class=LiveTagAPIArgs, data=data)