import argparse
from dataclasses import dataclass
import json
import os

@dataclass
class RuntimeConfig:
    output_string: str = "Processed"

def main(files, config: RuntimeConfig):
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    os.makedirs(out_path, exist_ok=True)
    for file in files:
        basename = os.path.basename(file)
        out = os.path.join(out_path, basename + "_tags.json")
        with open(file, 'r') as f:
            data = json.load(f)
            start_time = data.get('start_time', 0)
            end_time = data.get('end_time', 0)

        with open(out, 'w') as f:
            f.write(json.dumps([{"start_time": start_time * 1000, "end_time": end_time * 1000, "text": config.output_string}], indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='*', type=str, help='Input file paths')
    parser.add_argument('--config', type=str, required=False, help='Runtime configuration JSON')
    
    args = parser.parse_args()
    
    files = args.file_paths
    config = json.loads(args.config) if args.config else {}
    runtime_config = RuntimeConfig(**config)
    main(files, runtime_config)