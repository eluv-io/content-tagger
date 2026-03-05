import json
import os
import uuid
from dataclasses import asdict

from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, QueueItem, ListJobArgs, UpdateJobRequest
from src.fetch.model import Scope


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

    def create_job(self, args: CreateQueueItem, auth: str) -> None:
        id = str(uuid.uuid4())
        self._write_job(id, {
            "id": id,
            "qid": args.qid,
            "status": "queued",
            "params": asdict(args.params),
            "status_details": asdict(args.status_details),
            "auth": auth,
        })

    def claim_job(self, qid: str, auth: str) -> bool:
        for job in self._all_jobs():
            if job["qid"] == qid and job["status"] == "queued":
                job["status"] = "running"
                self._write_job(job["id"], job)
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
            p = job["params"]
            params = TagArgs(
                feature=p["feature"],
                run_config=p["run_config"],
                scope=Scope(**p["scope"]),
                replace=p["replace"],
                destination_qid=p["destination_qid"],
                max_fetch_retries=p["max_fetch_retries"],
            )
            results.append(QueueItem(
                id=job["id"],
                qid=job["qid"],
                params=params,
                auth=job["auth"],
                user=job["user"],
                tenant=job["tenant"],
            ))
        return results

    def update_job(self, args: UpdateJobRequest, auth: str) -> None:
        job = self._read_job(args.id)
        job["status"] = args.status
        job["status_details"] = asdict(args.status_details)
        self._write_job(args.id, job)

    def stop_job(self, id: str, auth: str) -> None:
        job = self._read_job(id)
        job["status"] = "cancelled"
        self._write_job(id, job)