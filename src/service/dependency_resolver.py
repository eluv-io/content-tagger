from dataclasses import dataclass

from src.common.content import Content
from src.common.model import ModelConfig
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.model import ListJobArgs, QueueItem, job_status
from src.tags.track_resolver import TrackResolver


@dataclass(frozen=True)
class JobDependencies:
    # ids of the jobs this job has to wait for
    parents: list[str]
    # id of a queued/running job with the same model and stream
    duplicate_of: str | None


class DependencyResolver:
    """
    Works out which jobs a batch of pending jobs must wait for, based on the tracks each model depends on and
    produces, and which of them duplicate a job that is already queued or running.
    """

    def __init__(
        self,
        job_store: JobStore,
        track_resolver: TrackResolver,
        model_configs: dict[str, ModelConfig],
    ):
        self.jobstore = job_store
        self.track_resolver = track_resolver
        self.model_configs = model_configs

    def resolve(self, q: Content, job_ids: list[str], args: list[TagArgs]) -> list[JobDependencies]:
        """Returns the dependencies of each job, where job_ids[i] is the pending job for args[i].

        A job depends on the jobs in the batch producing its dependency tracks, or failing that on the queued,
        running or pending jobs producing them. Dependents of a duplicate wait on the job it duplicates.
        """
        active = self._list_jobs(q, "running") + self._list_jobs(q, "queued")
        active_by_stream = {(item.model, item.params.scope.get_stream()): item.id for item in active if item.params}

        # job each arg resolves to: its own pending job, or the job it duplicates
        effective_ids: list[str] = []
        duplicate_of: list[str | None] = []
        batch_by_stream: dict[tuple[str, str], str] = {}
        for job_id, arg in zip(job_ids, args):
            key = (arg.feature, arg.scope.get_stream())
            dup = active_by_stream.get(key) or batch_by_stream.get(key)
            batch_by_stream.setdefault(key, job_id)
            duplicate_of.append(dup)
            effective_ids.append(dup or job_id)

        batch_by_track: dict[str, list[int]] = {}
        for i, arg in enumerate(args):
            for track in self.track_resolver.resolve(arg.feature):
                batch_by_track.setdefault(track.name, []).append(i)

        existing_by_track = self._existing_jobs_by_track(active + self._list_jobs(q, "pending"))

        res = []
        for i, arg in enumerate(args):
            if duplicate_of[i] is not None:
                res.append(JobDependencies(parents=[], duplicate_of=duplicate_of[i]))
                continue
            parents: list[str] = []
            for t in self.model_configs[arg.feature].track_dependencies:
                if t in batch_by_track:
                    parents.extend(effective_ids[idx] for idx in batch_by_track[t])
                elif t in existing_by_track:
                    parents.append(existing_by_track[t])
            res.append(JobDependencies(parents=list(dict.fromkeys(parents)), duplicate_of=None))
        return res

    def _list_jobs(self, q: Content, status: job_status) -> list[QueueItem]:
        return self.jobstore.list_jobs(ListJobArgs(qid=q.qid, status=status, include_unready=True), auth=q.token)

    def _existing_jobs_by_track(self, items: list[QueueItem]) -> dict[str, str]:
        """Returns a map of track name -> id of a job producing it."""
        result: dict[str, str] = {}
        for item in items:
            if item.model not in self.model_configs:
                continue
            for track in self.track_resolver.resolve(item.model):
                result.setdefault(track.name, item.id)
        return result
