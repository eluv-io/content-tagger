from podman import PodmanClient
from typing import List
import os
import sys

from config import config

# Run a container with the given feature and files
# Outputs list of tag files
def run_container(client: PodmanClient, feature: str, files: List[str]) -> List[str]:
    out_path = os.path.join(config["storage"]["tmp"], feature)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    volumes = [
        {
            "source": os.path.join(config["storage"]["tmp"], feature),
            # convention for containers to store tags in /elv/tags
            "target": "/elv/tags",
            "type": "bind",
        },
        {
            "source": config["storage"]["cache"],
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
    container = client.containers.create(image=feature,
                                command=[f"{os.path.basename(p)}" for p in files],  
                                mounts=volumes,  
                                remove=True,  
                                network_mode="host",  
                                devices=config["devices"], 
    )
    container.start()
    for log in container.logs(stream=True, stderr=True, stdout=True):
        sys.stderr.write(log.decode("utf-8"))
    tags = [os.path.join(config["storage"]["tmp"], feature, f"{os.path.basename(f)}_tags.json") for f in files]
    if config["services"][feature].get("frame_level", False):
        frame_tags = [os.path.join(config["storage"]["tmp"], feature, f"{os.path.basename(f)}_frametags.json") for f in files]
        tags += frame_tags
    return tags