from copy import deepcopy
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from dacite import from_dict

from src.common.errors import MissingResourceError
from src.status.get_info import UserInfoResolver
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.dto import TagDetailsRaw
from src.tagging.fabric_tagging.queue.model import *
from src.fetch.model import *

def _convert_scope(data: dict) -> Scope:
    type = data.get("type")
    if type == "processor":
        return TimeRangeScope(**data)
    elif type == "assets":
        return AssetScope(**data)
    elif type == "video":
        return VideoScope(**data)
    elif type == "livestream":
        return LiveScope(**data)
    elif type == "tag-aligned":
        return TagAlignedScope(**data)
    else:
        raise ValueError(f"Unknown scope type: {type}")

class FsJobStore:
    """Job store backed by one json file per job. All jobs are held in memory and files are write-through,
    so this process must be the only writer to store_dir."""

    def __init__(
        self, 
        store_dir: str,
        user_info_resolver: UserInfoResolver,
    ):
        self.store_dir = store_dir
        self.user_info_resolver = user_info_resolver
        # serializes read-modify-write on job files: API threads and the TagRunner
        # poll thread mutate the same job concurrently
        self._lock = threading.Lock()
        os.makedirs(store_dir, exist_ok=True)
        self._jobs = self._load_jobs()

    def _job_path(self, id: str) -> str:
        return os.path.join(self.store_dir, f"{id}.json")

    def _load_jobs(self) -> dict[str, dict]:
        jobs = {}
        for fname in os.listdir(self.store_dir):
            if fname.endswith(".json"):
                with open(os.path.join(self.store_dir, fname), "r") as f:
                    jobs[fname[:-5]] = json.load(f)
        return jobs

    def _read_job(self, id: str) -> dict:
        """Returns the cached job dict. Callers must not mutate it."""
        job = self._jobs.get(id)
        if job is None:
            raise MissingResourceError(f"Job {id} not found")
        return job

    def _write_job(self, id: str, data: dict) -> None:
        path = self._job_path(id)
        # temp name must be unique per write, otherwise concurrent writers
        # interleave into the same file and corrupt the job
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)  # atomic on Linux
        self._jobs[id] = data

    def _all_jobs(self) -> list[dict]:
        with self._lock:
            return list(self._jobs.values())
    
    def _convert_job_dict(self, job: dict) -> QueueItem:
        job = deepcopy(job)
        p = job["params"]
        params = TagArgs(
            feature=p["feature"],
            run_config=p["run_config"],
            scope=_convert_scope(p["scope"]),
            track_suffix=p.get("track_suffix", ""),
            replace=p["replace"],
            destination_qid=p["destination_qid"],
            index_qid=p.get("index_qid", ""),
            max_fetch_retries=p["max_fetch_retries"],
            caller_info=p.get("caller_info", {})
        )
        return QueueItem(
            id=job["id"],
            qid=job["qid"],
            params=params,
            created_at=job["created_at"],
            status=job["status"],
            status_details=from_dict(TagDetailsRaw, job["status_details"]).to_model() if job["status_details"] else None,
            error=job.get("error"),
            stop_requested=job["stop_requested"],
            auth=job["auth"],
            user=job["user"],
            tenant=job["tenant"],
            deps=job.get("deps", []),
            additional_info=job.get("additional_info", {}),
        )

    def create_job(self, args: CreateQueueItem, auth: str) -> QueueItem:
        id = str(uuid.uuid4())
        tenant = self.user_info_resolver.get_tenant(args.qid, auth)
        user_info = self.user_info_resolver.get_user_info(auth=auth, tenant_id=None)
        job = {
            "id": id,
            "qid": args.qid,
            "status": "queued",
            "created_at": time.time(),
            "params": asdict(args.params),
            "status_details": asdict(args.status_details) if args.status_details else None,
            "error": None,
            "stop_requested": False,
            "user": user_info.user_adr,
            "tenant": tenant,
            "auth": auth,
            "additional_info": args.additional_info,
            "deps": args.deps,
        }
        with self._lock:
            self._write_job(id, deepcopy(job))
        return self._convert_job_dict(job)

    def claim_job(self, id: str, auth: str) -> bool:
        with self._lock:
            job = deepcopy(self._read_job(id))
            if job["status"] == "queued":
                job["status"] = "running"
                self._write_job(id, job)
                return True
            return False

    def get_job(self, id: str) -> QueueItem:
        return self._convert_job_dict(self._read_job(id))

    def list_jobs(self, args: ListJobArgs, auth: str) -> list[QueueItem]:
        results = []
        for job in self._all_jobs():
            if args.status != "deleted" and job["status"] == "deleted":
                continue
            if args.qid and job["qid"] != args.qid:
                continue
            if args.user and job["user"] != args.user:
                continue
            if args.tenant and job["tenant"] != args.tenant:
                continue
            if args.status and job["status"] != args.status:
                continue
            # filter if unmet dependencies
            if job["status"] == "queued" and not args.include_unready:
                unmet_deps = [dep for dep in job.get("deps", []) if self._read_job(dep)["status"] in ("running", "queued")]
                if unmet_deps:
                    continue
            results.append(self._convert_job_dict(job))
        return results

    def update_job(self, args: UpdateJobRequest, auth: str) -> None:
        with self._lock:
            job = deepcopy(self._read_job(args.id))
            if job["status"] == "deleted" and args.status != "deleted":
                # a late status update must not resurrect a deleted job
                return
            job["status"] = args.status
            if args.status_details is not None:
                job["status_details"] = asdict(args.status_details)
            if args.error is not None:
                job["error"] = args.error
            self._write_job(args.id, job)

    def stop_job(self, id: str, auth: str) -> None:
        with self._lock:
            job = deepcopy(self._read_job(id))
            job["stop_requested"] = True
            self._write_job(id, job)