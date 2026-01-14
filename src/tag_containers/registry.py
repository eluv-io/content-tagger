import os
import uuid
from datetime import datetime
from copy import deepcopy
from podman import PodmanClient
from loguru import logger

from src.common.errors import MissingResourceError
from src.tag_containers.model import RegistryConfig, ContainerSpec, ModelConfig
from src.tag_containers.containers import *
from src.tag_containers.model import ContainerRequest

logger = logger.bind(name="Container Registry")

class ContainerRegistry:
    """
    Get runnable containers through identifier
    """

    def __init__(self, cfg: RegistryConfig):
        self.pclient = PodmanClient()
        self.cfg = cfg
        os.makedirs(self.cfg.base_dir, exist_ok=True)
        os.makedirs(self.cfg.cache_dir, exist_ok=True)

    def get(self, req: ContainerRequest) -> TagContainer:
        if req.job_id is not None:
            jobid = req.job_id
        else:
            jobid = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + str(uuid.uuid4())[:6]
            logger.warning(f"User request {req} did not give jobid, generating default: {jobid}")

        jobpath = os.path.join(self.cfg.base_dir, req.model_id, jobid)
        tags_path = os.path.join(jobpath, 'tags')
        logs_path = os.path.join(jobpath, 'log.out')

        cache_path = self.cfg.cache_dir

        modelcfg = self.cfg.model_configs.get(req.model_id)
        if not modelcfg:
            raise MissingResourceError(f"Model {req.model_id} not found")

        ccfg = ContainerSpec(
            id=jobid,
            media_input=req.media_input,
            run_config=req.run_config,
            logs_path=logs_path,
            cache_dir=cache_path,
            tags_dir=tags_path,
            model_config=modelcfg,
        )

        if req.live:
            return LiveTagContainer(self.pclient, ccfg)
        else:    
            return TagContainer(self.pclient, ccfg)

    def get_model_config(self, model: str) -> ModelConfig:
        if model not in self.cfg.model_configs:
            raise BadRequestError(f"Model {model} not found")
        return deepcopy(self.cfg.model_configs[model])

    def services(self) -> list[str]:
        """
        Returns a list of available services
        """
        # TODO: check if the image exists
        return list(self.cfg.model_configs.keys())