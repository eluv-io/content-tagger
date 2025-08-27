
from dataclasses import dataclass
from common_ml.types import Data
from src.fetch.types import AssetScope, VideoScope
from src.tagger.fabric_tagging.types import RunConfig, TagArgs

@dataclass
class TagAPIArgs(Data):
    # maps feature name to RunConfig
    features: dict[str, RunConfig]
    # start_time in milliseconds (defaults to 0)
    start_time: int | None=None
    # end_time in milliseconds (defaults to entire content)
    end_time: int | None=None
    # replace tag files if they already exist
    replace: bool=False

    @staticmethod
    def from_dict(data: dict) -> 'TagAPIArgs':
        features = {feature: RunConfig(**cfg) for feature, cfg in data['features'].items()}
        return TagAPIArgs(features=features, start_time=data.get('start_time', None), end_time=data.get('end_time', None), replace=data.get('replace', False))

    def to_tag_args(self) -> TagArgs:
        return TagArgs(
            features=self.features,
            scope=VideoScope(start_time=self.start_time, end_time=self.end_time),
            replace=self.replace
        )


@dataclass
class ImageTagAPIArgs(Data):
    # maps feature name to RunConfig
    features: dict[str, RunConfig]

    # asset file paths to tag relative to the content object e.g. /assets/image.jpg, if empty then we will look in /meta/assets and tag all the image assets located there. 
    assets: list[str] | None

    # replace tag files if they already exist
    replace: bool=False

    @staticmethod
    def from_dict(data: dict) -> 'ImageTagAPIArgs':
        features = {feature: RunConfig(stream='image', **cfg) for feature, cfg in data['features'].items()}
        return ImageTagAPIArgs(features=features, assets=data.get('assets', None), replace=data.get('replace', False))

    def to_tag_args(self) -> TagArgs:
        return TagArgs(
            features=self.features,
            scope=AssetScope(assets=self.assets),
            replace=self.replace
        )
