import json

import requests
from dateutil import parser

from src.common.errors import BadRequestError, JobConflictError, MissingResourceError
from src.common.logging import logger
from src.tagging.fabric_tagging.queue.common import job_uid
from src.tagging.fabric_tagging.queue.model import *
from src.tagging.fabric_tagging.queue.serialize import details_from_dict, details_to_dict, params_from_dict, params_to_dict, system_message

logger = logger.bind(name="Queue Manager")

PAGE_SIZE = 100

class _Conflict(Exception):
    """The job's state doesn't allow the request (409)."""

class QueueManagerJobStore:
    """Job store backed by the queue manager service."""

    def __init__(self, base_url: str, worker_secret: str | None, timeout: int, job_type: str = "tag"):
        self.base_url = base_url.rstrip('/')
        self.worker_secret = worker_secret
        self.timeout = timeout
        self.job_type = job_type
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def _log_response_and_raise(self, response: requests.Response):
        """Log response content before raising HTTPError"""
        try:
            response_json = response.json()
            logger.error(f"{json.dumps(response_json)}")
        except Exception:
            logger.error(f"HTTP {response.status_code} response (non-JSON): {response.text}")
        response.raise_for_status()

    def _request(self, method: str, path: str, as_user: str | None = None, **kwargs) -> dict:
        """Make request with the appropriate headers depending on whether it's a worker endpoint or not."""
        if as_user is not None:
            headers = {'Authorization': f"Bearer {as_user}"}
        else:
            headers = {'X-Worker-Secret': self.worker_secret or ""}
        response = self.session.request(method, f"{self.base_url}{path}", headers=headers, timeout=self.timeout, **kwargs)
        if response.status_code == 404:
            raise MissingResourceError(f"{method} {path} not found")
        if response.status_code == 409:
            raise _Conflict(f"{method} {path}: {response.text}")
        if not response.ok:
            self._log_response_and_raise(response)
        return response.json()

    def _to_item(self, job: dict) -> QueueItem:
        """Convert job API response to internal struct"""
        details = job.get("status_details")
        # a job completed while cancelling also gets a system_message, which isn't an error
        error = job.get("error") or (system_message(details) if job["status"] == "cancelled" else None)
        return QueueItem(
            id=job["id"],
            qid=job["qid"],
            model=job.get("subtype", ""),
            created_at=parser.isoparse(job["created_at"]).timestamp(),
            params=params_from_dict(job.get("params")),
            status=job["status"],
            status_details=details_from_dict(details),
            error=error,
            # only the /worker API returns auth; the job type has auth_storage "authorization", so it is the
            # submitter's Authorization header
            auth=(job.get("auth") or {}).get("Authorization", "").removeprefix("Bearer "),
            user=job.get("user", ""),
            tenant=job.get("tenant", ""),
            additional_info=job.get("additional_info") or {},
        )

    def create_job(self, args: CreateQueueItem, auth: str) -> QueueItem:
        try:
            job = self._request("POST", f"/q/{args.qid}/jobs", as_user=auth, json={
                "type": self.job_type,
                "subtype": args.model,
                "hold": True,
                "resource_hash": job_uid(args.qid, args.model),
                "enforce_resource": "active",
            })
        except _Conflict:
            raise JobConflictError(f"A job is already active for {args.model} on {args.qid}")
        return self._to_item(job)

    def release_job(self, args: ReleaseJobRequest, auth: str) -> bool:
        try:
            self._request("POST", f"/jobs/{args.id}/release", json={
                "params": params_to_dict(args.params),
                "dependencies": args.deps,
                "additional_info": args.additional_info,
            })
        except _Conflict:
            # no longer pending
            return False
        return True

    def claim_job(self, id: str, auth: str) -> bool:
        try:
            self._request("POST", f"/worker/jobs/{id}/claim")
        except _Conflict:
            # another job with the same resource is running
            return False
        except requests.HTTPError as e:
            if e.response.status_code == 400:
                # not queued
                return False
            raise
        return True

    def get_job(self, id: str) -> QueueItem:
        job = self._request("GET", f"/jobs/{id}")
        if job.get("archived"):
            raise MissingResourceError(f"Job {id} was deleted")
        return self._to_item(job)

    def list_jobs(self, args: ListJobArgs, auth: str) -> list[QueueItem]:
        query = {
            "type": self.job_type,
            "status": args.status,
            "qid": args.qid,
            "user": args.user,
            "tenant": args.tenant,
            "include_unready": args.include_unready,
            "ignore_resource_hash": args.include_unready,
        }
        query = {k: v for k, v in query.items() if v is not None}
        items: list[QueueItem] = []
        while args.limit is None or len(items) < args.limit:
            limit = PAGE_SIZE if args.limit is None else min(PAGE_SIZE, args.limit - len(items))
            page = self._request("GET", "/worker/jobs", params={**query, "start": len(items), "limit": limit})
            jobs = page.get("jobs") or []
            items.extend(self._to_item(job) for job in jobs)
            if not jobs or len(items) >= page["meta"]["total"]:
                break
        return items

    def update_progress(self, id: str, status_details: TagDetails, auth: str) -> QueueItem:
        try:
            job = self._request("PATCH", f"/worker/jobs/{id}", json={"status_details": details_to_dict(status_details)})
        except _Conflict:
            # the job has ended
            return self.get_job(id)
        return self._to_item(job)

    def complete_job(self, args: CompleteJobRequest, auth: str) -> bool:
        body: dict = {"status": args.status}
        if args.status_details is not None:
            body["status_details"] = details_to_dict(args.status_details)
        if args.error is not None:
            body["error"] = args.error
        try:
            self._request("POST", f"/worker/jobs/{args.id}/complete", json=body)
        except _Conflict:
            return False
        return True

    def cancel_job(self, id: str, auth: str, reason: str | None = None) -> None:
        try:
            job = self._request("POST", f"/jobs/{id}/cancel", as_user=auth)
        except _Conflict:
            # already ended or cancelling
            return
        if reason is not None and job["status"] == "cancelled":
            # a body with only status_details records it on a cancelled job
            self._request("POST", f"/worker/jobs/{id}/complete", json={"status_details": {"system_message": reason}})

    def delete_job(self, id: str, auth: str) -> None:
        try:
            self._request("DELETE", f"/jobs/{id}", as_user=auth)
        except _Conflict:
            raise BadRequestError(f"Job {id} hasn't ended")
