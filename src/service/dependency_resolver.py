from src.common.content import Content
from src.common.model import ModelConfig
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.model import ListJobArgs, QueueItem, job_status
from src.tags.track_resolver import TrackResolver


class DependencyResolver:
    """
    Works out which jobs a batch of pending jobs must wait for, based on the tracks each model depends on and
    produces.
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

    def resolve(self, q: Content, job_ids: list[str], args: list[TagArgs]) -> list[list[str]]:
        """Returns the ids of the jobs each job must wait for, where job_ids[i] is the pending job for args[i].

        A job depends on the jobs in the batch producing its dependency tracks, or failing that on the queued,
        running or pending jobs producing them.
        """
        batch_by_track: dict[str, list[str]] = {}
        for job_id, arg in zip(job_ids, args):
            for track in self.track_resolver.resolve(arg.feature):
                batch_by_track.setdefault(track.name, []).append(job_id)

        existing = self._list_jobs(q, "running") + self._list_jobs(q, "queued") + self._list_jobs(q, "pending")
        existing_by_track = self._existing_jobs_by_track(existing)

        res = []
        for arg in args:
            parents: list[str] = []
            for t in self.model_configs[arg.feature].track_dependencies:
                if t in batch_by_track:
                    parents.extend(batch_by_track[t])
                elif t in existing_by_track:
                    parents.append(existing_by_track[t])
            res.append(list(dict.fromkeys(parents)))
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
