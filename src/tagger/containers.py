
from copy import deepcopy
from dataclasses import dataclass
from podman import PodmanClient
from loguru import logger
import json
import os
import psutil
from typing import Literal

from src.api.errors import MissingResourceError
from src.tagger.resource_manager import SystemResources

@dataclass
class ContainerSpec:
    image: str
    cachepath: str
    logspath: str
    tagspath: str
    fileargs: list[str]
    runconfig: dict

class TagContainer:

    def __init__(
        self,
        pclient: PodmanClient,
        cfg: ContainerSpec
    ):
        self.cfg = cfg
        self.pclient = pclient
        self.container = None

    def start(
        self, 
        gpuidx: int | None,
    ) -> None:

        volumes = [
            {
                "source": self.cfg.tagspath,
                # convention for containers to store tags in /elv/tags
                "target": "/elv/tags",
                "type": "bind",
            },
            {
                "source": self.cfg.cachepath,
                # convention for python modules to store cache in /root/.cache
                "target": "/root/.cache",
                "type": "bind",
                "read_only": False
            }
        ]

        for f in self.cfg.fileargs:
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

        kwargs = {
            "image": self.cfg.image,
            "command": [f"{os.path.basename(f)}" for f in self.cfg.fileargs] + ["--config", f"{json.dumps(self.cfg.runconfig)}"],
            "mounts": volumes,
            "remove": True,
            "network_mode": "host",
            "log_config": {
                "type": "file",
                "path": self.cfg.logspath,
            }
        }

        if gpuidx is not None:
            kwargs["devices"] = [f"nvidia.com/gpu={gpuidx}"]

        container = self.pclient.containers.create(**kwargs)
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

    def tags(self) -> list[str]:
        """
        Get set of files currently open for writing by this container
        """
        if not self.is_running():
            return []

        assert self.container is not None

        try:
            # Get the container's main process PID
            container_info = self.container.inspect()
            pid = container_info.get("State", {}).get("Pid")
            
            if not pid:
                return []

            # Get the process and its open files
            process = psutil.Process(pid)
            open_files = []
            for open_file in process.open_files():
                open_files.append(open_file.path)
            
            return open_files
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            # If we can't get process info, assume all files might be in use
            return []
        
@dataclass
class ModelConfig:
    name: str
    image: str
    type: Literal["video", "audio", "frame"]
    resources: SystemResources

@dataclass
class RegistryConfig:
    modconfigs: dict[str, ModelConfig]
    logspath: str
    tagspath: str
    cachepath: str

class ContainerRegistry:
    """
    Get runnable containers through identifier
    """

    def __init__(self, cfg: RegistryConfig):
        self.pclient = PodmanClient()
        self.cfg = cfg
        os.makedirs(self.cfg.logspath, exist_ok=True)
        os.makedirs(self.cfg.tagspath, exist_ok=True)
        os.makedirs(self.cfg.cachepath, exist_ok=True)

    def get(self, model: str, fileargs: list[str], runconfig: dict) -> TagContainer:
        tagspath = os.path.join(self.cfg.tagspath, model)
        logspath = os.path.join(self.cfg.logspath, model)
        cachepath = os.path.join(self.cfg.cachepath, model)

        modelcfg = self.cfg.registry.get(model)
        if not modelcfg:
            raise MissingResourceError(f"Model {model} not found")

        ccfg = ContainerSpec(
            image=modelcfg.image,
            fileargs=fileargs,
            runconfig=runconfig,
            logspath=logspath,
            cachepath=cachepath,
            tagspath=tagspath
        )

        return TagContainer(self.pclient, ccfg)

    def get_model_resources(self, model: str) -> SystemResources:
        return deepcopy(self.cfg.registry[model].resources)

    def services(self) -> list[str]:
        """
        Returns a list of available services
        """
        # TODO: check if the image exists
        return list(self.cfg.registry.keys())