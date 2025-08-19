
from dataclasses import dataclass
from podman import PodmanClient
from podman.domain.containers import Container
from loguru import logger
import json
import os
from typing import List

@dataclass
class ContainerSpec:
    image: str
    volumes: list[dict]
    command: list[str]
    logfile: str

class TagContainer:
    def __init__(self, cspec: ContainerSpec):
        self.cspec = cspec
        self.container: Container | None = None

    def start(self, pclient: PodmanClient, gpu_idx: int | None) -> None:
        kwargs = {
            "image": self.cspec.image,
            "command": self.cspec.command,
            "mounts": self.cspec.volumes,
            "remove": True,
            "network_mode": "host",
            "log_config": {
                "type": "file",
                "path": self.cspec.logfile,
            }
        }
        if gpu_idx is not None:
            kwargs["devices"] = [f"nvidia.com/gpu={gpu_idx}"]
        container = pclient.containers.create(**kwargs)
        container.start()
        self.container = container

    def stop(self) -> None:
        if not self.container:
            return
        if self.container.status == "running":
            # podman client will kill if it doesn't stop within the timeout limit
            self.container.stop(timeout=5)
        self.container.reload()
        if self.container.status == "running":
            logger.error(f"Container status is still \"running\" after stop. Please check the container and stop it manually.")

    def is_running(self) -> bool:
        if self.container is None:
            return False
        self.container.reload()
        return self.container.status == "running" or self.container.status == "created"

class PodmanConfig:
    client: PodmanClient
    cachepath: str

def list_images(pclient: PodmanClient) -> List[str]:
    with pclient as podman_client:
        return sum([image.tags for image in podman_client.images.list() if image.tags], [])

def create_container(
        cfg: PodmanConfig,
        # name of the image to use
        image: str,
        save_path: str,
        fileargs: list[str],
        config: dict,
        logfile: str
    ) -> ContainerSpec:
    """
    Creates a tagger container
    """

    os.makedirs(save_path, exist_ok=True)

    volumes = [
        {
            "source": save_path,
            # convention for containers to store tags in /elv/tags
            "target": "/elv/tags",
            "type": "bind",
        },
        {
            "source": cfg.cachepath,
            # convention for python modules to store cache in /root/.cache
            "target": "/root/.cache",
            "type": "bind",
            "read_only": False
        }
    ]

    for f in fileargs:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File {f} not found")
        elif not os.path.isfile(f):
            raise IsADirectoryError(f"{f} is a directory")
        elif not os.path.isabs(f):
            raise ValueError(f"{f} must be an absolute path")
        # mount the file
        volumes.append({
            "source": f,
            "target": f"/elv/{os.path.basename(f)}",
            "type": "bind",
            "read_only": True
        })
    return ContainerSpec(
        image=image,
        command=[f"{os.path.basename(f)}" for f in fileargs] + ["--config", f"{json.dumps(config)}"],
        volumes=volumes,
        logfile=logfile
    )