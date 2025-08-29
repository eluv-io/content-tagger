import argparse
import json
import os
import sys
from typing import List

from common_ml.model import FrameModel, default_tag
from common_ml.model import FrameTag
from common_ml.utils import nested_update

class DummyModel(FrameModel):
    def __init__(self, run_config: dict):
        self.config = run_config
        self.tags = self.config["tags"]
        self.idx = 0

    def get_config(self):
        return self.config
    
    def set_config(self, config):
        self.config = config

    def tag(self, frame) -> List[FrameTag]:
        tag = self.tags[self.idx]
        self.idx = (self.idx + 1) % len(self.tags)
        return [FrameTag(text=tag, box={"x1": 0, "y1": 0, "x2": 0, "y2": 0}, confidence=1.0)]

def run(files, runtime_config):
    print('fooballs')
    print('fooballs', file=sys.stderr)
    default_cfg = {
        "fps": 1,
        "allow_single_frame": True,
        "tags": ["a", "b", "b", "c"]
    }
    if runtime_config is None:
        cfg = default_cfg
    else:
        cfg = json.loads(runtime_config)
        cfg = nested_update(default_cfg, cfg)
    model = DummyModel(run_config=cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    print(files)
    default_tag(model, files, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.file_paths, args.config)