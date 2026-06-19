

from functools import lru_cache
from src.common.logging.timing import timeit
from src.common.logging import logger

from src.common.content import Content, QAPIFactory
from src.common.model import ModelConfig
from src.fetch.model import Scope
from src.service.model import TagStartResult
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs, QueueItem
from src.tags.track_resolver import TrackResolver


class JobPoster:
    """
    JobPoster is responsible for posting tagging jobs to the JobStore, taking dependencies into account
    """

    def __init__(
        self, 
        job_store: JobStore,
        track_resolver: TrackResolver,
        model_configs: dict[str, ModelConfig],
        qfactory: QAPIFactory
    ):
        self.jobstore = job_store
        self.track_resolver = track_resolver
        self.model_configs = model_configs
        self.qfactory = qfactory
    
    def post_jobs(self, q: Content, args: list[TagArgs]) -> list[TagStartResult]:
        res: list[TagStartResult | None] = [None for _ in args]

        # maps arg idx -> list of arg idx dependencies
        deps: dict[int, list[int]] = {}

        # maps arg idx -> list of dependents
        dependents: dict[int, list[int]] = {}

        # maps arg idx -> list of external job ids it depends on (jobs not in this request)
        external_deps: dict[int, list[str]] = {}

        track_to_arg_idx: dict[str, list[int]] = {}
        for i, arg in enumerate(args):
            output_tracks = self.track_resolver.resolve(arg.feature)
            for track in output_tracks:
                track_to_arg_idx.setdefault(track.name, []).append(i)

        existing_jobs_by_track = self._get_existing_jobs_by_track(q)

        for i, arg in enumerate(args):
            dep_tracks = self.model_configs[arg.feature].track_dependencies
            for t in dep_tracks:
                in_request = track_to_arg_idx.get(t, [])
                if in_request:
                    for idx in in_request:
                        deps.setdefault(i, []).append(idx)
                        dependents.setdefault(idx, []).append(i)
                elif t in existing_jobs_by_track:
                    external_deps.setdefault(i, []).append(existing_jobs_by_track[t])

        jobs_submitted = 0

        num_dependencies = {i: len(deps.get(i, [])) for i in range(len(args))}

        # job id per arg idx
        job_ids = ["" for _ in args]

        # loop through over and over till all are satisfied
        while jobs_submitted < len(args):
            for i in range(len(args)):
                if job_ids[i] != "":
                    # already posted
                    continue

                if num_dependencies[i] == 0:
                    # see if there is a job already running
                    existing_job = self._get_already_running(q, args[i].feature, args[i].scope)

                    if not existing_job:

                        parents = [job_ids[idx] for idx in deps.get(i, [])] + external_deps.get(i, [])

                        job = self._post_job(q, args[i], parents)

                        job_ids[i] = job.id

                        res[i] = TagStartResult(started=True, created_at=job.created_at, job_id=job.id, dependencies=parents, message="Job enqueued")

                    else:
                        # mark the job id of already running job so we can set it as a dependency
                        job_ids[i] = existing_job.id

                        res[i] = TagStartResult(started=False, created_at=existing_job.created_at, job_id=existing_job.id, dependencies=[], message="Job already running")

                    # resolve dependents
                    for d in dependents.get(i, []):
                        num_dependencies[d] -= 1

                    jobs_submitted += 1

        type_checked_res = []
        
        # placate type checker
        for r in res:
            assert isinstance(r, TagStartResult)
            type_checked_res.append(r)

        return type_checked_res
    
    def _get_existing_jobs_by_track(self, q: Content) -> dict[str, str]:
        """Returns a map of track name -> job id for tracks produced by running/queued jobs."""
        result: dict[str, str] = {}
        running = self.jobstore.list_jobs(ListJobArgs(qid=q.qid, status="running", include_unready=True), auth=q.token)
        queued = self.jobstore.list_jobs(ListJobArgs(qid=q.qid, status="queued", include_unready=True), auth=q.token)
        for item in list(running) + list(queued):
            feature = item.params.feature
            if feature not in self.model_configs:
                continue
            for track in self.track_resolver.resolve(feature):
                result.setdefault(track.name, item.id)
        return result

    def _get_already_running(self, q: Content, model: str, scope: Scope) -> QueueItem | None:
        running = self.jobstore.list_jobs(ListJobArgs(qid=q.qid, status="running", include_unready=True), auth=q.token)
        for item in running:
            item_stream = item.params.scope.get_stream()
            if item.params.feature == model and item_stream == scope.get_stream():
                return item
        queued = self.jobstore.list_jobs(ListJobArgs(qid=q.qid, status="queued", include_unready=True), auth=q.token)
        for item in queued:
            item_stream = item.params.scope.get_stream()
            if item.params.feature == model and item_stream == scope.get_stream():
                return item
        return None

    def _post_job(self, q: Content, arg: TagArgs, deps: list[str]) -> QueueItem:
        with timeit("getting display title", min_duration=2):
            title = self._get_display_title(q)

        with timeit("creating job", min_duration=1):
            return self.jobstore.create_job(
                CreateQueueItem(
                    qid=q.qid,
                    params=arg,
                    status_details=None,
                    deps=deps,
                    additional_info={"title": title},
                ),
                auth=q.token,
            )

    @lru_cache(maxsize=1024)
    def _get_display_title(self, q: Content) -> str:
        qapi = self.qfactory.create(q)
        title = qapi.content_object_metadata(metadata_subtree="/public/name")
        if not isinstance(title, str):
            return ""
        return title