import yaml
import os
from typing import Any

def load_config() -> Any:   
    root_path = os.path.dirname(os.path.abspath(__file__))
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    # If the path is not absolute, we assume it is relative to the root directory
    for p in config["storage"].keys():
        if not config["storage"][p].startswith('/'):
            config["storage"][p] = os.path.join(root_path, config["storage"][p])
        if not os.path.exists(config["storage"][p]):
            os.makedirs(config["storage"][p])

    for service in config["services"]:
        os.makedirs(os.path.join(config["storage"]["tags"], service), exist_ok=True)
        
    return config

config = load_config()