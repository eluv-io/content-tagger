import argparse
from dataclasses import dataclass
import json
import os
import sys
import time

@dataclass
class RuntimeConfig:
    output_string: str = "Processed"
    sleep_hack: int = 2

def main(files, config: RuntimeConfig):
    print(f"sleeping for a bit to simulate processing time for {config.sleep_hack} seconds.", file=sys.stderr)
    time.sleep(config.sleep_hack)
    print("Done sleeping, writing tag files now", file=sys.stderr)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    os.makedirs(out_path, exist_ok=True)
    for file in files:
        basename = os.path.basename(file)
        out = os.path.join(out_path, basename + "_tags.json")
        if not os.path.exists(out):        
            with open(file, 'r') as f:
                data = json.load(f)
                start_time = data.get('start_time', 0)
                end_time = data.get('end_time', 0)
            with open(out, 'w') as f:
                f.write(json.dumps([{"start_time": start_time, "end_time": end_time, "text": config.output_string}], indent=4))
        else:
            print(f"File {file} already existed, skipping", file=sys.stderr)
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='*', type=str, help='Input file paths')
    parser.add_argument('--config', type=str, required=False, help='Runtime configuration JSON')
    
    args = parser.parse_args()
    
    files = args.file_paths
    config = json.loads(args.config) if args.config else {}
    runtime_config = RuntimeConfig(**config)
    main(files, runtime_config)
