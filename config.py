import yaml
import os
from typing import Any

def load_config(path = None) -> Any:
    root_path = os.getcwd()

    if path is None:
        path = os.path.dirname(os.path.abspath(__file__)) + '/config.yml'

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # If the path is not absolute, we assume it is relative to the current directory
    for p in config["storage"].keys():
        if not config["storage"][p].startswith('/'):
            config["storage"][p] = os.path.join(root_path, config["storage"][p])
        if not os.path.exists(config["storage"][p]):
            os.makedirs(config["storage"][p])

    for modelname, modelconf in config["services"].items():
        if len(modelconf.get("cpu_slots", [])) > 0:
            assert modelconf.get("allowed_gpus", None) == [], f"If cpuslots are set, allowed_gpus must be empty. Check {modelname} in config.yml"
        elif modelconf.get("allowed_gpus", None) == []:
            # set the default cpu slots
            modelconf["cpu_slots"] = config["devices"]["default_cpu_slots"][:]

    return config

config = load_config()

def reload_config(file: str):
    newconfig = load_config(file)
    for k in list(config.keys()):
        del config[k]
    for k, v in newconfig.items():
        config[k] = v
        