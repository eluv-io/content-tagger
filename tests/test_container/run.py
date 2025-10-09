import argparse
import json
import os
import sys
from typing import List

from common_ml.model import FrameModel, default_tag, run_live_mode
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

def get_runtime_config(runtime_config: str | None = None) -> dict:
    """Get the runtime configuration, merging with defaults if provided"""
    default_cfg = {
        "fps": 1,
        "allow_single_frame": True,
        "tags": ["a", "b", "b", "c"]
    }
    
    if runtime_config is None:
        return default_cfg
    else:
        cfg = json.loads(runtime_config)
        return nested_update(default_cfg, cfg)

def run(file_paths: List[str], runtime_config: str | None = None):
    """Generate tag files from a list of video/image files and a runtime config"""
    print('Starting test_model run', file=sys.stderr)
    
    # Print received file paths (for testing)
    for filepath in file_paths:
        print(f"Got {filepath}")
        sys.stdout.flush()
    
    cfg = get_runtime_config(runtime_config)
    model = DummyModel(run_config=cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    default_tag(model, file_paths, out_path)

def get_tag_fn(runtime_config: str | None = None):
    """Create a tag function with the specified configuration"""
    cfg = get_runtime_config(runtime_config)
    model = DummyModel(run_config=cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    
    def tag_fn(file_paths: List[str]):
        # Print received file paths (for testing)
        for filepath in file_paths:
            print(f"Got {filepath}")
            sys.stdout.flush()
        
        default_tag(model, file_paths, out_path)
    
    return tag_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='*', type=str, help='Input file paths')
    parser.add_argument('--config', type=str, required=False, help='Runtime configuration JSON')
    parser.add_argument('--live', action='store_true', help='Run in live mode (read files from stdin)')
    
    args = parser.parse_args()
    
    if args.live:
        print('Running in live mode', file=sys.stderr)
        tag_fn = get_tag_fn(args.config)
        run_live_mode(tag_fn)
    else:
        if not args.file_paths:
            print("Error: No file paths provided", file=sys.stderr)
            sys.exit(1)
        print('Running in batch mode', file=sys.stderr)
        run(args.file_paths, args.config)