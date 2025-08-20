import threading
from collections import defaultdict
import time
import uuid

from loguru import logger

from src.tagger.model_containers.containers import ContainerRegistry
from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tagger.fabric_tagging.types import Job, JobArgs, JobStatus, JobState, JobStore, JobID
from src.common.content import Content
from src.api.tagging.format import TagArgs, ImageTagArgs
from src.common.errors import MissingResourceError
from src.fetch.fetch_video import Fetcher
from src.tags.tagstore import FilesystemTagStore, Job as TagJob

class FabricTagger:

    """
    Handles the flow of downloading data from fabric, tagging, and uploading
    """

    def __init__(
            self, 
            manager: SystemTagger,
            cregistry: ContainerRegistry,
            tagstore: FilesystemTagStore,
            fetcher: Fetcher
        ):

        self.manager = manager
        self.cregistry = cregistry
        self.tagstore = tagstore
        self.fetcher = fetcher

        self.dblock = threading.Lock()

        self.storelock = threading.Lock()
        self.jobstore = JobStore()

        # map qid -> token
        self.tokens = {}
        
        self.shutdown_signal = threading.Event()

    def _validate_args(self, args: TagArgs | ImageTagArgs) -> None:
        """
        Raises error if args are bad
        """

        services = self.cregistry.services()

        modconfigs = self.cregistry.cfg.modconfigs

        for feature in args.features:
            if feature not in services:
                raise MissingResourceError(
                    f"Invalid feature: {feature}. Available features: {', '.join(services)}"
                )

            if isinstance(args, TagArgs):
                continue

            modeltype = modconfigs[feature].type

            if modeltype != "frame":
                raise MissingResourceError(
                    f"{feature} is not frame-level"
                )

    def tag(self, q: Content, args: TagArgs | ImageTagArgs) -> dict:
        self._validate_args(args)
        # TODO: handle image
        if not isinstance(args, TagArgs):
            raise NotImplementedError("Image tagging is not implemented yet")
        status = {}
        for feature in args.features:
            job = Job(
                args=JobArgs(
                    q=q,
                    feature=feature,
                    runconfig=args.features[feature],
                    start_time=args.start_time,
                    end_time=args.end_time,
                ),
                status=JobStatus.starting(),
                taghandle="",
                lock=threading.Lock(),
                stopevent=threading.Event(),
                container=None
            )
            err = self._start_job(job)
            if err:
                status[feature] = err
            else:
                status[feature] = "Job started successfully"
        return status

    def _start_job(self, job: Job) -> str:
        with self.storelock:
            jobid = job.get_id()

            if jobid in self.jobstore.active_jobs:
                return f"Job {jobid} is already running"

            self.jobstore.active_jobs[jobid] = job

            if isinstance(job.args, ImageTagArgs):
                return "Image tagging is not implemented yet"
            else:
                threading.Thread(target=self._run_job, args=(job, )).start()

        return ""

    def _stop_job(self, jobid: JobID, status: JobState, error: Exception | None) -> None:
        logger.exception(error)
        with self.storelock:
            if jobid in self.jobstore.active_jobs:
                job = self.jobstore.active_jobs[jobid]

                job.status.status = status
                job.status.time_ended = time.time()

                self.jobstore.inactive_jobs[jobid] = job
                del self.jobstore.active_jobs[jobid]
            else:
                logger.warning(f"Tried to stop an inactive job: {jobid}")

    def _run_job(self, job: Job) -> None:
        jobid = job.get_id()

        tsjob = TagJob(
            id=str(uuid.uuid4()),
            qhit=job.args.q.qhit,
            feature=job.args.feature,
            stream=job.args.runconfig.stream,
            timestamp=time.time(),
            author="tagger"
        )

        self.tagstore.start_job(tsjob)

        # 1. download
        job.status.status = "Fetching content"
        try:
            media_files, failed = self._download_content(job, start_time=job.args.start_time, end_time=job.args.end_time)
        except Exception as e:
            self._stop_job(jobid, "Failed", e)
            return

        if job.stopevent.is_set():
            return

        job.status.failed += failed

        # 2. tag
        with self.storelock: # TODO: better way?
            container = self.cregistry.get(job.args.feature, media_files, job.args.runconfig.model)
            reqresources = self.cregistry.get_model_resources(job.args.feature)
            taggingdone = threading.Event()
            uid = self.manager.start(container, reqresources, taggingdone)
            job.container = container
            job.taghandle = uid
            job.status.status = "Tagging content"

        taggingdone.wait()

        if job.stopevent.is_set():
            return

        # 3. upload
        self._stop_job(jobid, "Completed", None)
        #self._upload_content(job, media_files)

    def status(self, qhit: str) -> dict[str, dict[str, JobStatus]]:
        """
        Args:
            qhit (str): content object, hash, or write token that files belong to
        Returns:
            dict[str, dict[str, JobStatus]]: a dictionary mapping stream -> feature -> JobStatus
        """

        active_jobs = self.jobstore.active_jobs
        inactive_jobs = self.jobstore.inactive_jobs

        with self.storelock:
            jobids = {jobid for jobid in active_jobs.keys() if jobid.qhit == qhit}
            jobids |= {jobid for jobid in inactive_jobs.keys() if jobid.qhit == qhit}

            if len(jobids) == 0:
                raise MissingResourceError(
                    f"No jobs started for {qhit}"
                )

            res = defaultdict(dict)
            for jid in jobids:
                job = active_jobs.get(jid, inactive_jobs[jid])
                stream, feature = job.args.runconfig.stream, job.args.feature
                res[stream][feature] = self._summarize_status(job.status)

        return dict(res)

    def _summarize_status(self, status: JobStatus) -> dict:
        end = time.time()
        if status.time_ended:
            end = status.time_ended
        return {
            "status": status.status,
            "time_running": end - status.time_started,
            "tagging_progress": status.tagging_progress,
            "failed": status.failed,
        }

    def stop(self, qhit: str, feature: str) -> None:
        active_jobs = self.jobstore.active_jobs
        with self.storelock:
            jobids = {jobid for jobid in active_jobs.keys() if jobid.qhit == qhit and jobid.feature == feature}

            if len(jobids) == 0:
                raise MissingResourceError(
                    f"No job running for {feature} on {qhit}"
                )

            jobs = [active_jobs[jid] for jid in jobids]
            for job in jobs:
                job.stopevent.set()

                if job.taghandle:
                    try:
                        self.manager.stop(job.taghandle)
                    except Exception as e:
                        logger.error(f"Error stopping job {job.get_id()}: {e}")

    def _job_watcher(self) -> None:
        while not self.shutdown_signal.is_set():
            try:
                with self.storelock:
                    self._check_jobs()
            except Exception as e:
                logger.exception(f"Unexpected error in job watcher: {e}")
            time.sleep(0.2)

    def _check_jobs(self) -> None:
        for jobid, job in self.jobstore.active_jobs.items():
            if not job.status == "Tagging content":
                continue

            self._upload_tags(job)

    def cleanup(self) -> None:
        self.shutdown_signal.set()
        self.manager.shutdown()

    def _copy_new_files(
            self, 
            job: Job,
        ) -> None:
        
                