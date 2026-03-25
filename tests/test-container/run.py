import argparse
import json
from typing import List
import time

from common_ml.tagging.models.frame_based import FrameModel
from common_ml.tagging.run_helpers import start_loop_from_frame_model
from common_ml.tagging.models.tag_types import FrameTag
from common_ml.utils.dictionary import nested_update

class DummyModel(FrameModel):
    def __init__(self, run_config: dict):
        self.config = run_config
        self.tags = self.config["tags"]
        self.idx = 0

    def get_config(self):
        return self.config
    
    def set_config(self, config):
        self.config = config

    def tag_frame(self, img) -> List[FrameTag]:
        time.sleep(self.config["delay"])
        tag = self.tags[self.idx]
        self.idx = (self.idx + 1) % len(self.tags)
        return [FrameTag(tag=tag, box={"x1": 0, "y1": 0, "x2": 0, "y2": 0})]

def get_runtime_config(runtime_config: str | None = None) -> dict:
    """Get the runtime configuration, merging with defaults if provided"""
    default_cfg = {
        "delay": 0,
        "fps": 1,
        "allow_single_frame": True,
        "tags": ["a", "b", "b", "c"]
    }
    
    if runtime_config is None:
        return default_cfg
    else:
        cfg = json.loads(runtime_config)
        return nested_update(default_cfg, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--params', type=str, required=False, help='Runtime configuration JSON')
    
    args = parser.parse_args()

    params = get_runtime_config(args.params)

    model = DummyModel(params)
    
    start_loop_from_frame_model(model, args.output_path)