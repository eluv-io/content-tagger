import json
import os
import time
import uuid
from dataclasses import asdict
from dacite import from_dict

from src.tagging.fabric_tagging.model import TagArgs
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
    else:
        raise ValueError(f"Unknown scope type: {type}")

class FsJobStore:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def _job_path(self, id: str) -> str:
        return os.path.join(self.store_dir, f"{id}.json")

    def _read_job(self, id: str) -> dict:
        with open(self._job_path(id), "r") as f:
            return json.load(f)

    def _write_job(self, id: str, data: dict) -> None:
        with open(self._job_path(id), "w") as f:
            json.dump(data, f, indent=2)

    def _all_jobs(self) -> list[dict]:
        jobs = []
        for fname in os.listdir(self.store_dir):
            if fname.endswith(".json"):
                id = fname[:-5]
                jobs.append(self._read_job(id))
        return jobs
    
    def _convert_job_dict(self, job: dict) -> QueueItem:
        p = job["params"]
        params = TagArgs(
            feature=p["feature"],
            run_config=p["run_config"],
            scope=_convert_scope(p["scope"]),
            replace=p["replace"],
            destination_qid=p["destination_qid"],
            max_fetch_retries=p["max_fetch_retries"],
        )
        return QueueItem(
            id=job["id"],
            qid=job["qid"],
            params=params,
            created_at=job["created_at"],
            status=job["status"],
            status_details=from_dict(JobStatus, job["status_details"]),
            stop_requested=job["stop_requested"],
            auth=job["auth"],
            user=job["user"],
            tenant=job["tenant"],
        )

    def create_job(self, args: CreateQueueItem, auth: str) -> QueueItem:
        id = str(uuid.uuid4())
        self._write_job(id, {
            "id": id,
            "qid": args.qid,
            "status": "queued",
            "created_at": time.time(),
            "params": asdict(args.params),
            "status_details": asdict(args.status_details),
            "stop_requested": False,
            "user": "",
            "tenant": "",
            "auth": auth,
        })
        job_data = self._read_job(id)
        return self._convert_job_dict(job_data)

    def claim_job(self, id: str, auth: str) -> bool:
        job = self._read_job(id)
        if job["status"] == "queued":
            job["status"] = "running"
            self._write_job(id, job)
            return True
        return False

    def list_jobs(self, args: ListJobArgs, auth: str) -> list[QueueItem]:
        results = []
        for job in self._all_jobs():
            if args.qid and job["qid"] != args.qid:
                continue
            if args.user and job["user"] != args.user:
                continue
            if args.tenant and job["tenant"] != args.tenant:
                continue
            if args.status and job["status"] != args.status:
                continue
            results.append(self._convert_job_dict(job))
        return results

    def update_job(self, args: UpdateJobRequest, auth: str) -> None:
        job = self._read_job(args.id)
        job["status"] = args.status
        job["status_details"] = asdict(args.status_details)
        self._write_job(args.id, job)

    def stop_job(self, id: str, auth: str) -> None:
        job = self._read_job(id)
        job["stop_requested"] = True
        self._write_job(id, job)