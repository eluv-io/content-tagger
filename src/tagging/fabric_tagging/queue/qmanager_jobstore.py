import json

import requests
from dateutil import parser

from src.common.errors import BadRequestError, MissingResourceError
from src.common.logging import logger
from src.tagging.fabric_tagging.queue.model import *
from src.tagging.fabric_tagging.queue.serialize import details_from_dict, details_to_dict, params_from_dict, params_to_dict, system_message

logger = logger.bind(name="Queue Manager")

JOB_TYPE = "tag"
PAGE_SIZE = 100
# the queue manager doesn't return a job's dependencies, so they are mirrored into additional_info at release
DEPS_KEY = "dependencies"

def _token(auth: dict | None) -> str:
    # the "tag" job type has auth_storage "authorization": the queue stores the submitter's Authorization header
    return (auth or {}).get("Authorization", "").removeprefix("Bearer ")

class QueueManagerJobStore:
    """Job store backed by the queue manager service. Jobs are created with type "tag" and the model as subtype.
    Submitting, cancelling and deleting act as the user, with their token; everything else acts as a worker, with
    the worker secret, which takes precedence over a token."""

    def __init__(self, base_url: str, worker_secret: str | None, timeout: int):
        self.base_url = base_url.rstrip('/')
        self.worker_secret = worker_secret
        self.timeout = timeout
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

    def _request(self, method: str, path: str, as_user: str | None = None, refused: tuple[int, ...] = (), **kwargs) -> dict | None:
        """Makes the request as the user with token as_user, or else as a worker. Returns the response body, or None
        if the queue manager answered with a status in refused."""
        if as_user is not None:
            headers = {'Authorization': f"Bearer {as_user}"}
        else:
            headers = {'X-Worker-Secret': self.worker_secret or ""}
        response = self.session.request(method, f"{self.base_url}{path}", headers=headers, timeout=self.timeout, **kwargs)
        if response.status_code == 404:
            raise MissingResourceError(f"{method} {path} not found")
        if response.status_code in refused:
            logger.debug(f"{method} {path} refused: {response.text}")
            return None
        if not response.ok:
            self._log_response_and_raise(response)
        return response.json()

    def _to_item(self, job: dict, auth: str = "") -> QueueItem:
        details = job.get("status_details")
        # a job completed while cancelling also gets a system_message, which isn't an error
        error = job.get("error") or (system_message(details) if job["status"] == "cancelled" else None)
        additional_info = dict(job.get("additional_info") or {})
        deps = additional_info.pop(DEPS_KEY, [])
        return QueueItem(
            id=job["id"],
            qid=job["qid"],
            model=job.get("subtype", ""),
            created_at=parser.isoparse(job["created_at"]).timestamp(),
            params=params_from_dict(job.get("params")),
            status=job["status"],
            status_details=details_from_dict(details),
            error=error,
            auth=_token(job.get("auth")) or auth,
            user=job.get("user", ""),
            tenant=job.get("tenant", ""),
            deps=deps,
            additional_info=additional_info,
        )

    def create_job(self, args: CreateQueueItem, auth: str) -> QueueItem:
        job = self._request("POST", f"/q/{args.qid}/jobs", as_user=auth, json={
            "type": JOB_TYPE,
            "subtype": args.model,
            "hold": True,
        })
        return self._to_item(job, auth=auth) # type: ignore

    def release_job(self, args: ReleaseJobRequest, auth: str) -> bool:
        # the release body can't set resource_hash
        if self._request("PATCH", f"/worker/jobs/{args.id}", refused=(409,), json={"resource_hash": args.resource}) is None:
            return False
        released = self._request("POST", f"/jobs/{args.id}/release", refused=(409,), json={
            "params": params_to_dict(args.params),
            "dependencies": args.deps,
            "additional_info": {**args.additional_info, DEPS_KEY: args.deps},
            "enforce_resource": "active",
        })
        return released is not None

    def claim_job(self, id: str, auth: str) -> bool:
        return self._request("POST", f"/worker/jobs/{id}/claim", refused=(400, 409)) is not None

    def get_job(self, id: str) -> QueueItem:
        job: dict = self._request("GET", f"/jobs/{id}") # type: ignore
        if job.get("archived"):
            raise MissingResourceError(f"Job {id} was deleted")
        return self._to_item(job)

    def list_jobs(self, args: ListJobArgs, auth: str) -> list[QueueItem]:
        query = {
            "type": JOB_TYPE,
            "status": args.status,
            "qid": args.qid,
            "user": args.user,
            "tenant": args.tenant,
            "include_unready": str(args.include_unready).lower(),
            "ignore_resource_hash": str(args.include_unready).lower(),
        }
        query = {k: v for k, v in query.items() if v is not None}
        items: list[QueueItem] = []
        while args.limit is None or len(items) < args.limit:
            limit = PAGE_SIZE if args.limit is None else min(PAGE_SIZE, args.limit - len(items))
            page: dict = self._request("GET", "/worker/jobs", params={**query, "start": len(items), "limit": limit}) # type: ignore
            jobs = page.get("jobs") or []
            items.extend(self._to_item(job) for job in jobs)
            if not jobs or len(items) >= page["meta"]["total"]:
                break
        return items

    def update_progress(self, id: str, status_details: TagDetails, auth: str) -> QueueItem:
        job = self._request("PATCH", f"/worker/jobs/{id}", refused=(409,), json={
            "status_details": details_to_dict(status_details),
        })
        if job is None:
            # the job has ended
            return self.get_job(id)
        return self._to_item(job)

    def complete_job(self, args: CompleteJobRequest, auth: str) -> bool:
        body: dict = {"status": args.status}
        if args.status_details is not None:
            body["status_details"] = details_to_dict(args.status_details)
        if args.error is not None:
            body["error"] = args.error
        return self._request("POST", f"/worker/jobs/{args.id}/complete", refused=(409,), json=body) is not None

    def cancel_job(self, id: str, auth: str, reason: str | None = None) -> None:
        job = self._request("POST", f"/jobs/{id}/cancel", as_user=auth, refused=(409,))
        if job is None or reason is None or job["status"] != "cancelled":
            return
        # a body with only status_details records it on a cancelled job
        self._request("POST", f"/worker/jobs/{id}/complete", refused=(409,), json={
            "status_details": {"system_message": reason},
        })

    def delete_job(self, id: str, auth: str) -> None:
        if self._request("DELETE", f"/jobs/{id}", as_user=auth, refused=(409,)) is None:
            raise BadRequestError(f"Job {id} hasn't ended")
