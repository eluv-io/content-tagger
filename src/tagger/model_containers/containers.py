
from copy import deepcopy
from podman import PodmanClient
from loguru import logger
import json
import os
import psutil

from src.api.errors import MissingResourceError
from src.tagger.resource_manager import SystemResources
from src.tagger.model_containers.types import ContainerSpec, RegistryConfig

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
        if self.is_running():
            logger.warning(f"Container {self.container.id} did not stop in time, killing it")
            self.container.kill()

    def is_running(self) -> bool:
        if self.container is None:
            return False
        self.container.reload()
        return self.container.status == "running" or self.container.status == "created"

    def tags(self) -> list[str]:
        """
        Get set of files currently open for writing by this container
        """

        tags = os.listdir(self.cfg.tagspath)

        if not self.is_running():
            return tags

        assert self.container is not None

        # Get the container's main process PID
        container_info = self.container.inspect()
        pid = container_info.get("State", {}).get("Pid")
        
        if not pid:
            return tags

        # Get the process and its open files
        process = psutil.Process(pid)
        for open_file in process.open_files():
            if open_file.path in tags:
                # TODO: check that it's full path.
                tags.remove(open_file.path)

        return tags

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

        modelcfg = self.cfg.modconfigs.get(model)
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
        return deepcopy(self.cfg.modconfigs[model].resources)

    def services(self) -> list[str]:
        """
        Returns a list of available services
        """
        # TODO: check if the image exists
        return list(self.cfg.modconfigs.keys())