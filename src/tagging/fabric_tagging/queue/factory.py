from src.status.get_info import UserInfoResolver
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.queue.fs_jobstore import FsJobStore
from src.tagging.fabric_tagging.queue.model import JobStoreConfig
from src.tagging.fabric_tagging.queue.qmanager_jobstore import QueueManagerJobStore

def create_jobstore(cfg: JobStoreConfig, user_info_resolver: UserInfoResolver) -> JobStore:
    if cfg.base_url:
        return QueueManagerJobStore(cfg.base_url, cfg.worker_secret, cfg.timeout, cfg.job_type)
    else:
        return FsJobStore(cfg.base_dir, user_info_resolver=user_info_resolver)
