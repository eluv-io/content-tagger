from podman import PodmanClient
from podman.domain.containers import Container
import json
import os
from typing import List, Optional
from loguru import logger

from config import config

def list_services() -> List[str]:
    with PodmanClient() as podman_client:
        images = sum([image.tags for image in podman_client.images.list() if image.tags], [])

    res = []
    for service in config['services']:
        if config['services'][service]['image'] in images:
            res.append(service)
        else:
            logger.error(f"Image {config['services'][service]['image']} not found")
    return res

# Run a container with the given feature and files
# Outputs list of tag files
def create_container(client: PodmanClient, feature: str, save_path: str, files: List[str], run_config: dict, device_idx: Optional[int], out: str="/dev/null") -> Container:
    os.makedirs(save_path, exist_ok=True)
    if len(files) == 0:
        raise ValueError("No files provided")
    volumes = [
        {
            "source": save_path,
            # convention for containers to store tags in /elv/tags
            "target": "/elv/tags",
            "type": "bind",
        },
        {
            "source": config["storage"]["container_cache"],
            # convention for python modules to store cache in /root/.cache
            "target": "/root/.cache",
            "type": "bind",
            "read_only": False
        }
    ]
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found")
        elif not os.path.isfile(file):
            raise IsADirectoryError(f"{file} is a directory")
        elif not os.path.isabs(file):
            raise ValueError(f"{file} must be an absolute path")
        # mount the file
        volumes.append({
            "source": file,
            "target": f"/elv/{os.path.basename(file)}",
            "type": "bind",
            "read_only": True
        })
    kwargs = {"image": config["services"][feature]["image"],
            "command": [f"{os.path.basename(p)}" for p in files] + ["--config", f"{json.dumps(run_config)}"], 
            "mounts": volumes, 
            "remove": True, 
            "network_mode": "host", 
        }
    if device_idx is not None:
        kwargs["devices"] = [f"nvidia.com/gpu={device_idx}"]
    container = client.containers.create(**kwargs)
    return container