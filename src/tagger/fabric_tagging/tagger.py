import threading
from collections import defaultdict
import time
import os
from typing import Literal
from dataclasses import dataclass, field
import shutil
import tempfile

from loguru import logger
from requests import HTTPError

from src.tagger.model_containers.containers import ContainerRegistry, TagContainer
from src.tagger.sytem_tagging.resource_manager import  SystemTagger
from src.fabric.content import Content
from src.api.tagging.format import TagArgs, ImageTagArgs
from src.api.errors import MissingResourceError
from src.fabric.fetch_video import download_stream
from src.fabric.fetch_assets import fetch_assets


class FabricTagger:

    """
    Handles the flow of downloading data from fabric, tagging, and uploading
    """

    def __init__(
            self, 
            manager: SystemTagger,
            cregistry: ContainerRegistry,
            tconfig: TaggerConfig
        ):

        self.manager = manager
        self.cregistry = cregistry
        self.tconfig = tconfig

        self.dblock = threading.Lock()

        self.storelock = threading.Lock()
        self.jobstore = JobStore()

        # maps (qhit, stream) -> lock
        self.download_lock = defaultdict(threading.Lock)

        # controls the number of concurrent downloads
        self.dl_sem = threading.Semaphore(tconfig.max_downloads)

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
            uuid = self.manager.start(container, reqresources, taggingdone)
            job.container = container
            job.taghandle = uuid
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

            self._copy_new_files(job)

    def cleanup(self) -> None:
        self.shutdown_signal.set()
        self.manager.shutdown()

    def _download_content(self, job: Job, **kwargs) -> tuple[list[str], list[str]]:
        media_files, failed = [], []
        stream = job.args.runconfig.stream
        qhit = job.args.q.qhit

        save_path = os.path.join(self.tconfig.partspath, qhit, stream)

        # TODO: if waiting for lock, and stop_event is set, it will keep waiting and stop only after the lock is acquired.

        with self.download_lock[(qhit, stream)]:
            with self.dl_sem:
                if stream == "image":
                    media_files, failed = fetch_assets(job.args.q, save_path,  **kwargs)
                else:
                    media_files, failed = download_stream(job.args.q, stream, save_path, **kwargs, exit_event=job.stopevent)

        if job.stopevent.is_set():
            return [], []

        return media_files, failed

    def _filter_tagged_media_files(
            self,
            media_files: list[str], 
            q: Content,
            stream: str, 
            feature: str
    ) -> list[str]:
        """
        Args:
            media_files (List[str]): list of media files to filter
            qhit (str): content object, hash, or write token that files belong to
            stream (str): stream name
            feature (str): model name

        Returns:
            List[str]: list of media files that have not been tagged, filtered subset of media_files
        """
        try:
            if stream == "image":
                tag_files = q.list_files(path=f"image_tags/{feature}")
            else:
                tag_files = q.list_files(path=f"video_tags/{stream}/{feature}")
        except HTTPError:
            # if the folder doesn't exist, then no files have been tagged
            return media_files[:]
        tagged = set(self._source_from_tag_file(tag) for tag in tag_files)
        untagged = []
        for media_file in media_files:
            filename = os.path.basename(media_file)
            if filename not in tagged:
                untagged.append(media_file)
        return untagged

    def _filter_tagged_files(
            self,
            tagfiles: list[str], 
            q: Content,
            stream: str, 
            feature: str
        ) -> list[str]:
        """
        Args:
            media_files (List[str]): list of media files to filter
            qhit (str): content object, hash, or write token that files belong to
            stream (str): stream name
            feature (str): model name

        Returns:
            List[str]: list of media files that have not been tagged, filtered subset of media_files
        """
        try:
            if stream == "image":
                remote_files = q.list_files(path=f"image_tags/{feature}")
            else:
                remote_files = q.list_files(path=f"video_tags/{stream}/{feature}")
        except HTTPError:
            # if the folder doesn't exist, then no files have been tagged
            return tagfiles[:]
        untagged = []
        for tagfile in tagfiles:
            if not os.path.basename(tagfile) in remote_files:
                untagged.append(tagfile)
        return untagged

    def _source_from_tag_file(self, tagged_file: str) -> str:
        """
        Args:
            tagged_file (str): a tag file name, generated by tagger

        Returns:
            str: the source file name that the tag file was generated from
        """
        if tagged_file.endswith("_imagetags.json"):
            return tagged_file.split("_imagetags.json")[0]
        if tagged_file.endswith("_frametags.json"):
            return tagged_file.split("_frametags.json")[0]
        if tagged_file.endswith("_tags.json"):
            return tagged_file.split("_tags.json")[0]

        raise ValueError(f"Unknown tag file format: {tagged_file}")

    def _copy_new_files(
            self, 
            job: Job,
        ) -> None:
        destpath = os.path.join(self.tconfig.tagspath, job.args.q.qhit, job.args.runconfig.stream, job.args.feature)
        os.makedirs(destpath, exist_ok=True)

        if not job.container:
            return

        tags = job.container.tags()


        for tag in tags:
            dest_file = os.path.join(destpath, os.path.basename(tag))
            
            if os.path.exists(dest_file):
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
                temp_path = temp_file.name
            
            try:
                shutil.copyfile(tag, temp_path)
                # Atomic move to final destination
                shutil.move(temp_path, dest_file)
            except Exception as e:
                logger.exception(f"Failed to copy {tag} to {dest_file}: {e}")
            finally:
                # Clean up temp file if copy failed
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass  # Best effort cleanup
                