

from functools import lru_cache
from src.common.logging.timing import timeit

from src.common.content import Content, QAPIFactory
from src.common.model import ModelConfig
from src.service.model import TagStartResult
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, QueueItem
from src.tags.track_resolver import TrackResolver
from tests.core_tagging.conftest import model_configs


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
        res = []

        # maps arg idx -> list of arg idx dependencies
        deps: dict[int, list[int]] = {}

        # maps arg idx -> list of dependents
        dependents: dict[int, list[int]] = {}

        track_to_arg_idx: dict[str, list[int]] = {}
        for i, arg in enumerate(args):
            output_tracks = self.track_resolver.resolve(arg.feature)
            for track in output_tracks:
                track_to_arg_idx.setdefault(track.name, []).append(i)

        for i, arg in enumerate(args):
            dep_tracks = self.model_configs[arg.feature].track_dependencies
            for t in dep_tracks:
                for idx in track_to_arg_idx.get(t, []):
                    deps.setdefault(i, []).append(idx)
                    dependents.setdefault(idx, []).append(i)

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
                    parents = [job_ids[idx] for idx in deps.get(i, [])]

                    job = self._post_job(q, args[i], parents)

                    job_ids[i] = job.id

                    # resolve dependents
                    for d in dependents.get(i, []):
                        num_dependencies[d] -= 1

                    res.append(TagStartResult(started=True, created_at=job.created_at, job_id=job.id, message="Job enqueued"))
                    jobs_submitted += 1

        return res

    def _post_job(self, q: Content, arg: TagArgs, deps: list[str]) -> QueueItem:
        with timeit("getting display title", min_duration=2):
            title = self._get_display_title(q)

        with timeit("creating job", min_duration=1):
            return self.jobstore.create_job(
                CreateQueueItem(
                    qid=q.qid,
                    params=arg,
                    status_details=None,
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