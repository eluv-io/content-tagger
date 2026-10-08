from copy import deepcopy
import json
import os
import threading
import time
import uuid

from src.common.errors import BadRequestError, JobConflictError, MissingResourceError
from src.status.get_info import UserInfoResolver
from src.tagging.fabric_tagging.queue.common import job_uid
from src.tagging.fabric_tagging.queue.model import *
from src.tagging.fabric_tagging.queue.serialize import details_from_dict, details_to_dict, params_from_dict, params_to_dict

# statuses each status may be completed as, matching the queue manager
_COMPLETIONS: dict[str, set[str]] = {
    "pending": {"succeeded", "failed"},
    "running": {"succeeded", "failed"},
    "cancelling": {"succeeded", "failed", "cancelled"},
}

_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

class FsJobStore:
    """Job store backed by one json file per job, following the queue manager's job lifecycle. All jobs are held
    in memory and files are write-through, so this process must be the only writer to store_dir."""

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
                    job = json.load(f)
                # soft-deleted by older versions of this store
                if job["status"] != "deleted":
                    jobs[fname[:-5]] = job
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

    def _set(self, id: str, **changes) -> None:
        job = deepcopy(self._read_job(id))
        job.update(changes)
        self._write_job(id, job)

    def _all_jobs(self) -> list[dict]:
        with self._lock:
            return list(self._jobs.values())

    def _is_ready(self, job: dict) -> bool:
        # deleted dependencies had ended, so they don't block
        return all(self._jobs[dep]["status"] == "succeeded" for dep in job.get("deps", []) if dep in self._jobs)

    def _uid_in_use(self, job: dict, statuses: set[str]) -> bool:
        uid = job.get("uid")
        return uid is not None and any(
            other["id"] != job["id"] and other.get("uid") == uid and other["status"] in statuses
            for other in self._jobs.values()
        )

    def _is_claimable(self, job: dict) -> bool:
        return self._is_ready(job) and not self._uid_in_use(job, {"running", "cancelling"})

    def _cascade(self, id: str, from_statuses: set[str], to_status: job_status) -> None:
        """Moves the jobs transitively depending on id that are in from_statuses to to_status."""
        parents = [id]
        while parents:
            parent = parents.pop()
            for job in list(self._jobs.values()):
                if parent in job.get("deps", []) and job["status"] in from_statuses:
                    self._set(job["id"], status=to_status, error=f"parent job {parent} {to_status}")
                    parents.append(job["id"])

    def _convert_job_dict(self, job: dict) -> QueueItem:
        job = deepcopy(job)
        params = params_from_dict(job["params"])
        return QueueItem(
            id=job["id"],
            qid=job["qid"],
            # jobs written before the model field existed only have it in params
            model=job.get("model") or params.feature, # type: ignore
            params=params,
            created_at=job["created_at"],
            status=job["status"],
            status_details=details_from_dict(job["status_details"]),
            error=job.get("error"),
            auth=job["auth"],
            user=job["user"],
            tenant=job["tenant"],
            additional_info=job.get("additional_info", {}),
        )

    def create_job(self, args: CreateQueueItem, auth: str) -> QueueItem:
        id = str(uuid.uuid4())
        tenant = self.user_info_resolver.get_tenant(args.qid, auth)
        user_info = self.user_info_resolver.get_user_info(auth=auth, tenant_id=None)
        job = {
            "id": id,
            "qid": args.qid,
            "model": args.model,
            "status": "pending",
            "created_at": time.time(),
            "params": None,
            "status_details": None,
            "error": None,
            "user": user_info.user_adr,
            "tenant": tenant,
            "auth": auth,
            "additional_info": {},
            "deps": [],
            "uid": job_uid(args.qid, args.model),
        }
        with self._lock:
            if self._uid_in_use(job, {"pending", "queued", "running", "cancelling"}):
                raise JobConflictError(f"A job is already active for {args.model} on {args.qid}")
            self._write_job(id, deepcopy(job))
        return self._convert_job_dict(job)

    def release_job(self, args: ReleaseJobRequest, auth: str) -> bool:
        with self._lock:
            job = deepcopy(self._read_job(args.id))
            if job["status"] != "pending":
                return False
            job["status"] = "queued"
            job["params"] = params_to_dict(args.params)
            job["deps"] = list(args.deps)
            job["additional_info"] = {**job["additional_info"], **deepcopy(args.additional_info)}
            self._write_job(args.id, job)
            return True

    def claim_job(self, id: str, auth: str) -> bool:
        with self._lock:
            job = self._read_job(id)
            if job["status"] != "queued" or not self._is_claimable(job):
                return False
            self._set(id, status="running")
            return True

    def get_job(self, id: str) -> QueueItem:
        return self._convert_job_dict(self._read_job(id))

    def list_jobs(self, args: ListJobArgs, auth: str) -> list[QueueItem]:
        results = []
        for job in self._all_jobs():
            if args.limit is not None and len(results) >= args.limit:
                break
            if args.qid and job["qid"] != args.qid:
                continue
            if args.user and job["user"] != args.user:
                continue
            if args.tenant and job["tenant"] != args.tenant:
                continue
            if args.status and job["status"] != args.status:
                continue
            if job["status"] == "queued" and not args.include_unready and not self._is_claimable(job):
                continue
            results.append(self._convert_job_dict(job))
        return results

    def update_progress(self, id: str, status_details: TagDetails, auth: str) -> QueueItem:
        with self._lock:
            if self._read_job(id)["status"] in ("pending", "running", "cancelling"):
                self._set(id, status_details=details_to_dict(status_details))
            return self._convert_job_dict(self._jobs[id])

    def complete_job(self, args: CompleteJobRequest, auth: str) -> bool:
        with self._lock:
            job = self._read_job(args.id)
            if args.status not in _COMPLETIONS.get(job["status"], set()):
                return False
            changes: dict = {"status": args.status}
            if args.status_details is not None:
                changes["status_details"] = details_to_dict(args.status_details)
            if args.error is not None:
                changes["error"] = args.error
            self._set(args.id, **changes)
            if args.status == "failed":
                self._cascade(args.id, {"pending", "queued", "running"}, "failed")
            return True

    def cancel_job(self, id: str, auth: str, reason: str | None = None) -> None:
        with self._lock:
            status = self._read_job(id)["status"]
            changes = {"error": reason} if reason else {}
            if status in ("pending", "queued"):
                self._set(id, status="cancelled", **changes)
            elif status == "running":
                self._set(id, status="cancelling", **changes)
            else:
                return
            self._cascade(id, {"pending", "queued"}, "cancelled")

    def delete_job(self, id: str, auth: str) -> None:
        with self._lock:
            if self._read_job(id)["status"] not in _TERMINAL_STATUSES:
                raise BadRequestError(f"Job {id} hasn't ended")
            os.remove(self._job_path(id))
            del self._jobs[id]
