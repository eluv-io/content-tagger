import os
import uuid
from datetime import datetime
from podman import PodmanClient
from loguru import logger

from src.tag_containers.model import RegistryConfig, ContainerSpec
from src.tag_containers.containers import *
from src.tag_containers.model import ContainerRequest

logger = logger.bind(name="Container Factory")

class ContainerFactory:
    """
    Builds runnable containers from requests
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
        output_path = os.path.join(jobpath, 'output.jsonl')
        logs_path = os.path.join(jobpath, 'log.out')

        cache_path = self.cfg.cache_dir

        ccfg = ContainerSpec(
            id=jobid,
            media_dir=req.media_dir,
            run_config=req.run_config,
            logs_path=logs_path,
            cache_dir=cache_path,
            output_path=output_path,
            model_config=req.model_config,
            q=req.q,
            index_qid=req.index_qid,
            env=self.cfg.env,
        )

        return TagContainer(self.pclient, ccfg)
