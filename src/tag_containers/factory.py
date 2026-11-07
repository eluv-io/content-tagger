import os
import uuid
from datetime import datetime
from cachetools.func import ttl_cache
from podman import PodmanClient
from loguru import logger
import requests

from src.common.http import raise_for_status
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
            env=self._env(),
        )

        return TagContainer(self.pclient, ccfg)

    def _env(self) -> dict[str, str]:
        env = dict(self.cfg.env)
        if "VERTICALIZE_URL" not in env:
            try:
                env["VERTICALIZE_URL"] = _discover_verticalize_service()
            except Exception as e:
                logger.opt(exception=e).warning("Failed to discover verticalize service")
        return env

@ttl_cache(maxsize=1, ttl=20)
def _discover_verticalize_service() -> str:
    """First vertical service node in the main net config, cached for 5 minutes."""
    NETWORK_CONFIG_URL = "https://main.net955305.contentfabric.io/config"
    resp = requests.get(NETWORK_CONFIG_URL, timeout=10)
    raise_for_status(resp)
    urls = resp.json()["network"]["services"].get("vertical")
    if not urls:
        raise ValueError(f"No vertical service nodes in {NETWORK_CONFIG_URL}")
    return urls[0]