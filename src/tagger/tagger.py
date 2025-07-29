import threading
from queue import Queue
from collections import defaultdict
import time
import json

from loguru import logger

from src.tagger.jobs import JobsStore
from src.tagger.resource_manager import ResourceManager
from src.fabric.content import Content
from src.api.tagging.format import TagArgs
from src.containers import list_services
from src.api.errors import MissingResourceError, BadRequestError
from src.tagger.jobs import Job

from config import config

class Tagger():
    def __init__(
            self, 
            job_store: JobsStore,
            manager: ResourceManager,
        ):
        self.job_store = job_store
        self.manager = manager

        self.cpu_queue = Queue()
        self.gpu_queue = Queue()

        self.filesystem_lock = threading.Lock()

        self.store_lock = threading.Lock()
        self.job_store = job_store

        # make sure no two streams are being downloaded at the same time. 
        # maps (qhit, stream) -> lock
        self.download_lock = defaultdict(threading.Lock)

        # controls the number of concurrent downloads
        self.dl_sem = threading.Semaphore(config["fabric"]["max_downloads"])
        
        self.shutdown_signal = threading.Event()

    def tag(self, qhit: str, q: Content, args: TagArgs):
        services = list_services()

        invalid_features = [feature for feature in args.features if feature not in services]

        if len(invalid_features) > 0:
            raise MissingResourceError(
                f"Invalid features: {', '.join(invalid_features)}. Available features: {', '.join(services)}"
            )
        
        with self.store_lock:
            if self.shutdown_signal.is_set():
                raise RuntimeError("Tagger is shutting down, cannot start new jobs")
            for feature, run_config in args.features.items():
                if run_config.stream is None:
                    # if stream name is not provided, we pick stream based on whether the model is audio/video based
                    run_config.stream = config["services"][feature]["type"]
                if (run_config.stream, feature) in self.job_store.active_jobs[q.qhit]:
                    raise BadRequestError(
                        f"{feature} tagging is already in progress for {q.qhit} on {run_config.stream}"
                    )
            for feature, run_config in args.features.items():
                # get the subset of GPUs that the model can run on, default to all of them
                allowed_gpus = config["services"][feature].get("allowed_gpus", list(range(self.manager.num_devices)))
                allowed_cpus = config["services"][feature].get("cpu_slots", [])
                logger.debug(f"Starting {feature} on {q.qid}: {run_config.stream}")
                job = Job(
                    q=q, 
                    run_config=run_config, 
                    feature=feature, 
                    media_files=[], 
                    failed=[], 
                    replace=args.replace, 
                    status="Starting", 
                    stop_event=threading.Event(), 
                    time_started=time.time(), 
                    allowed_gpus=allowed_gpus, 
                    allowed_cpus=allowed_cpus
                )
                self.job_store.active_jobs[q.qhit][(run_config.stream, feature)] = job
                threading.Thread(target=_video_tag, args=(job, _get_authorization(request), args.start_time, args.end_time)).start()
            return Response(response=json.dumps({'message': f'Tagging started on {qhit}'}), status=200, mimetype='application/json')
    
    def _video_tag(self, job: Job, start_time: int | None, end_time: int | None -> None:
        media_files, failed = self._download_content(job, q, start_time=start_time, end_time=end_time)
        job.media_files = media_files
        job.failed = failed
        _submit_tag_job(job, ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=authorization))

        def _download_content(job: Job, elv_client: ElvClient, **kwargs) -> List[str]:
        media_files, failed = [], []
        stream = job.run_config.stream
        qhit = job.qhit

        if stream == "image":
            save_path = os.path.join(config["storage"]["images"], qhit, stream)
        else:
            save_path = os.path.join(config["storage"]["parts"], qhit, stream)

        try:
            # TODO: if waiting for lock, and stop_event is set, it will keep waiting and stop only after the lock is acquired.
            with download_lock[(qhit, stream)]:
                job.status = "Fetching content"
                with dl_sem:
                    # if fetching finished while waiting for lock, this will return immediately
                    if stream == "image":
                        media_files, failed = fetch_assets(qhit, save_path, elv_client, **kwargs)
                    else:
                        media_files, failed =  download_stream(qhit, stream, save_path, elv_client, **kwargs, exit_event=job.stop_event)
            logger.debug(f"got list of media files {media_files}")
        except (StreamNotFoundError, AssetsNotFoundException):
            with lock:
                _set_stop_status(job, "Failed", f"Content for stream {stream} was not found for {qhit}")
        except HTTPError as e:
            with lock:
                _set_stop_status(job, "Failed", f"Failed to fetch stream {stream} for {qhit}: {str(e)}. Make sure authorization token hasn't expired.")
        except Exception as e:
            with lock:
                _set_stop_status(job, "Failed", f"Unknown error occurred while fetching stream {stream} for {qhit}: {str(e)}")
        if _check_exit(job):
            return [], []
        return media_files, failed

    def _submit_tag_job(job: Job, client: ElvClient) -> None:
        if _check_exit(job):
            return
        media_files = job.media_files
        if len(media_files) == 0:
            with lock:
                if _is_job_stopped(job):
                    return
                _set_stop_status(job, "Failed", f"No media files found for {job.qhit}")
            return
        total_media_files = len(media_files)
        if not job.replace:
            media_files = _filter_tagged_files(media_files, client, job.qhit, job.run_config.stream, job.feature)
        logger.debug(f"Tag status for {job.qhit}: {job.feature} on {job.run_config.stream}")
        logger.debug(f"Total media files: {total_media_files}, Media files to tag: {len(media_files)}, Media files already tagged: {total_media_files - len(media_files)}")
        if len(media_files) == 0:
            with lock:
                if _is_job_stopped(job):
                    return
                _set_stop_status(job, "Completed", f"Tagging already complete for {job.feature} on {job.qhit}")
            return
        if len(job.allowed_gpus) > 0:
            # model will run on a gpu
            with lock:
                # TODO: Probably don't need lock
                job.media_files = media_files
                job.status = "Waiting to be assigned GPU"
            gpu_queue.put(job)
        else:
            with lock:
                job.media_files = media_files
                job.status = "Waiting for CPU resource"
            cpu_queue.put(job)