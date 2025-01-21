import argparse
from typing import List, Optional, Literal, Dict, Iterable, Tuple
from flask import Flask, request, Response, Request
from flask_cors import CORS
from podman import PodmanClient
import json
from loguru import logger
from dataclasses import dataclass, asdict, field
import os
from elv_client_py import ElvClient
from queue import Queue
from collections import defaultdict
import threading
from requests.exceptions import HTTPError
import time
import shutil
import signal
import atexit
from common_ml.types import Data

from config import config
from src.fetch import fetch_stream, StreamNotFoundError, fetch_assets, AssetsNotFoundException
from src.manager import ResourceManager, NoGPUAvailable

# RunConfig gives model level tagging params
@dataclass
class RunConfig():
    # model config, used to overwrite the model level config
    model: dict=field(default_factory=dict) 
    # stream name to run the model on, None to use the default stream. "image" is a special case which will tag image assets
    stream: Optional[str]=None

@dataclass
class Job:
    status: Literal["Starting", 
                    "Fetching content",
                    "Waiting to be assigned GPU", 
                    "Completed", 
                    "Failed", 
                    "Stopped"]
    qid: str
    feature: str
    run_config: RunConfig
    stop_event: threading.Event
    media_files: List[str]
    time_started: float
    time_ended: Optional[float]=None
    # tag_job_id is the job id returned by the manager, will be None until the tagging starts (status is "Running")
    tag_job_id: Optional[str]=None
    error: Optional[str]=None
    
def get_flask_app():
    app = Flask(__name__)
    # manages the gpu and inference jobs
    manager = ResourceManager()
    
    # queue for jobs waiting to be assigned a GPU
    tag_queue = Queue()

    # locks active, and inactive jobs dictionaries
    lock = threading.Lock()
    # maps qid -> (feature, stream) -> job
    active_jobs = defaultdict(dict)
    inactive_jobs = defaultdict(dict)

    # make sure no two streams are being downloaded at the same time. 
    # maps (qid, stream) -> lock
    download_lock = defaultdict(threading.Lock)

    @app.route('/list', methods=['GET'])
    def list_services() -> Response:
        res = _list_services()    
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
    
    # TagArgs represents the request body for the /tag endpoint
    @dataclass
    class TagArgs(Data):
        # maps feature name to RunConfig
        features: Dict[str, RunConfig]
        # start_time in milliseconds (defaults to 0)
        start_time: Optional[int]=None
        # end_time in milliseconds (defaults to entire content)
        end_time: Optional[int]=None

        @staticmethod
        def from_dict(data: dict) -> 'TagArgs':
            features = {feature: RunConfig(**cfg) for feature, cfg in data['features'].items()}
            return TagArgs(features=features, start_time=data.get('start_time', None), end_time=data.get('end_time', None))

    @app.route('/<qid>/tag', methods=['POST'])
    def tag(qid: str) -> Response:
        try:
            args = TagArgs.from_dict(request.json)
        except TypeError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        
        client, error_response = _get_client(request, qid, config["fabric"]["parts_url"])
        if error_response:
            return error_response
        
        invalid_services = _get_invalid_features(args.features.keys())
        if invalid_services:
            return Response(response=json.dumps({'error': f"Services {invalid_services} not found"}), status=404, mimetype='application/json')
            
        with lock:
            for feature, run_config in args.features.items():
                if run_config.stream is None:
                    # if stream name is not provided, we pick stream based on whether the model is audio/video based
                    run_config.stream = config["services"][feature]["type"]
                if (run_config.stream, feature) in active_jobs[qid]:
                    return Response(response=json.dumps({'error': f"{feature} tagging is already in progress for {qid} on {run_config.stream}"}), status=400, mimetype='application/json')
            for feature, run_config in args.features.items():
                logger.debug(f"Starting {feature} on {qid}: {run_config.stream}")
                job = Job(qid=qid, run_config=run_config, feature=feature, media_files=[], status="Starting", stop_event=threading.Event(), time_started=time.time())
                active_jobs[qid][(run_config.stream, feature)] = job
                threading.Thread(target=_video_tag, args=(job, client, args.start_time, args.end_time)).start()
        return Response(response=json.dumps({'message': f'Tagging started on {qid}'}), status=200, mimetype='application/json')
    
    def _video_tag(job: Job, elv_client: ElvClient, start_time: Optional[int], end_time: Optional[int]) -> None:
        media_files = _download_content(job, elv_client, start_time=start_time, end_time=end_time)
        if _check_exit(job):
            return
        if len(media_files) == 0:
            with lock:
                if _is_job_stopped(job):
                    return
                _set_stop_status(job, "Failed", f"No media files found for {job.qid}")
            return
        with lock:
            job.media_files = media_files
            job.status = "Waiting to be assigned GPU"
        tag_queue.put(job)

     # TagArgs represents the request body for the /tag endpoint
    @dataclass
    class ImageTagArgs(Data):
        # maps feature name to RunConfig
        features: Dict[str, RunConfig]

        # asset file paths to tag relative to the content object e.g. /assets/image.jpg, if empty then we will look in /meta/assets and tag all the image assets located there. 
        assets: Optional[List[str]]

        # if true, will overwrite existing tags. 
        # TODO: currently ignored
        replace: bool

        @staticmethod
        def from_dict(data: dict) -> 'ImageTagArgs':
            features = {feature: RunConfig(stream='image', **cfg) for feature, cfg in data['features'].items()}
            return ImageTagArgs(features=features, assets=data.get('assets', None), replace=data.get('replace', False))
        
    @app.route('/<qid>/image_tag', methods=['POST'])
    def image_tag(qid: str) -> Response:
        try:
            args = ImageTagArgs.from_dict(request.json)
        except TypeError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        
        client, error_response = _get_client(request, qid, config["fabric"]["parts_url"])
        if error_response:
            return error_response
        
        invalid_services = _get_invalid_features(args.features.keys())
        if invalid_services:
            return Response(response=json.dumps({'error': f"Services {invalid_services} not found"}), status=404, mimetype='application/json')
            
        with lock:
            for feature in args.features.keys():
                if active_jobs[qid].get(('image', feature), None):
                    return Response(response=json.dumps({'error': f"Image tagging for at least one of the requested features, {feature}, is already in progress for {qid}"}), status=400, mimetype='application/json')
            for feature, run_config in args.features.items():
                job = Job(qid=qid, feature=feature, run_config=run_config, media_files=[], status="Starting", stop_event=threading.Event(), time_started=time.time())
                active_jobs[qid][('image', feature)] = job
                threading.Thread(target=_image_tag, args=(job, client, args.assets, args.replace)).start()
        return Response(response=json.dumps({'message': f'Image asset tagging started on {qid}'}), status=200, mimetype='application/json')
    
    def _image_tag(job: Job, elv_client: ElvClient, assets: Optional[List[str]], replace: bool) -> None:
        images = _download_content(job, elv_client, assets=assets)
        if _check_exit(job):
            return
        if len(images) == 0:
            with lock:
                if _is_job_stopped(job):
                    return
                _set_stop_status(job, "Failed", f"No media files found for {job.qid}")
            return
        with lock:
            job.media_files = images
            job.status = "Waiting to be assigned GPU"
        tag_queue.put(job)
    
    # Download content and tag
    def _download_content(job: Job, elv_client: ElvClient, **kwargs) -> List[str]:
        media_files = []
        stream = job.run_config.stream
        qid = job.qid

        if stream == "image":
            save_path = os.path.join(config["storage"]["images"], qid, stream)
        else:
            save_path = os.path.join(config["storage"]["parts"], qid, stream)

        try:
            # TODO: if waiting for lock, and stop_event is set, it will keep waiting and stop only after the lock is acquired.
            with download_lock[(qid, stream)]:
                job.status = "Fetching content"

                # if fetching finished while waiting for lock, this will return immediately
                if stream == "image":
                    media_files = fetch_assets(qid, save_path, elv_client, **kwargs, exit_event=job.stop_event)
                else:
                    media_files =  fetch_stream(qid, stream, save_path, elv_client, **kwargs, exit_event=job.stop_event)
        except (StreamNotFoundError, AssetsNotFoundException):
            with lock:
                _set_stop_status(job, "Failed", f"Content for stream {stream} was not found for {qid}")
        except HTTPError as e:
            with lock:
                _set_stop_status(job, "Failed", f"Failed to fetch stream {stream} for {qid}: {str(e)}. Make sure authorization token hasn't expired.")
        if _check_exit(job):
            return []
        return media_files

    def _job_starter():
        """
        Processes jobs in the tag_queue. Waits for GPU to be available and then starts the job.
        """
        while True:
            # blocks until a job is available
            job = tag_queue.get()
            if _check_exit(job):
                continue
            stopped = False
            while not manager.await_gpu(timeout=config["devices"]["wait_for_gpu_sleep"]):
                # check if the job has been stopped, if not then we go back to waiting for a GPU
                if _check_exit(job):
                    stopped = True
                    break
            if stopped:
                continue
            try:
                job_id = manager.run(job.feature, job.run_config.model, job.media_files)
                with lock:
                    if _is_job_stopped(job):
                        # if the job has been stopped while the container was starting
                        manager.stop(job.tag_job_id)
                        continue
                    job.tag_job_id = job_id
                    job.status = "Tagging content"
                logger.success(f"Started running {job.feature} on {job.qid}")
            except NoGPUAvailable:
                with lock:
                    _set_stop_status(job, "Failed", "Unexpected error when trying to run the model. No GPU available.")
            except Exception as e:
                with lock:
                    _set_stop_status(job, "Failed", str(e))
                    
    def _job_watcher() -> None:
        while True:
            for qid in active_jobs:
                for (stream, feature), job in list(active_jobs[qid].items()):
                    if not job.status == "Tagging content":
                        continue
                    if job.stop_event.is_set():
                        manager.stop(job.tag_job_id)
                        with lock:
                            _set_stop_status(job, "Stopped")
                        continue
                    status = manager.status(job.tag_job_id)
                    if status.status == "Running":
                        continue
                    # otherwise the job has finished: either successfully or with an error
                    if status.status == "Completed":
                        logger.success(f"Finished running {job.feature} on {job.qid}")
                        # move outputted tags to their correct place
                        _move_files(qid, job.run_config.stream, job.feature, status.tags)
                    job.status = status.status 
                    job.time_ended = time.time()
                    with lock:
                        # move job to inactive_jobs
                        inactive_jobs[qid][(stream, feature)] = job
                        del active_jobs[qid][(stream, feature)]
            time.sleep(config["watcher"]["sleep"])
            
    def _check_exit(job: Job) -> bool:
        """
        Returns True if the job has received a stop signal, False otherwise. 
        Also, sets the status of the job to "Stopped" if it has been stopped.

        Args:
            job (Job): Job to check

        Returns:
            bool: True if the job has been stopped, False
        """
        if job.stop_event.is_set():
            with lock:
                if _is_job_stopped(job):
                    return True
                _set_stop_status(job, "Stopped")
            return True
        return False

    def _move_files(qid: str, stream: str, feature: str, tags: List[str]) -> None:
        for tag in tags:
            os.makedirs(os.path.join(config["storage"]["tags"], qid, stream, feature), exist_ok=True)
            shutil.move(tag, os.path.join(config["storage"]["tags"], qid, stream, feature, os.path.basename(tag)))

    @dataclass
    class FinalizeArgs:
        write_token: str
        force: bool=False
        authorization: Optional[str]=None
    
    @app.route('/<qid>/finalize', methods=['POST'])
    def finalize(qid: str) -> Response:
        client, error_response = _get_client(request, qid, config["fabric"]["config_url"])
        if error_response:
            return error_response
        try:
            args = FinalizeArgs(**request.args)
        except TypeError as e:
            return Response(response=json.dumps({'message': 'invalid request', 'error': str(e)}), status=400, mimetype='application/json')
        qwt = args.write_token
        qlib = client.content_object_library_id(qid)
        with lock:
            jobs_running = sum(len(active_jobs[qid][feature]) for feature in active_jobs[qid])
        if jobs_running > 0 and not args.force:
            return Response(response=json.dumps({'error': 'Some jobs are still running. Use `force=true` to finalize anyway.'}), status=400, mimetype='application/json')

        file_jobs = []
        for stream in os.listdir(os.path.join(config["storage"]["tags"], qid)):
            for feature in os.listdir(os.path.join(config["storage"]["tags"], qid, stream)):
                for tag in os.listdir(os.path.join(config["storage"]["tags"], qid, stream, feature)):
                    file_jobs.append(ElvClient.FileJob(local_path=os.path.join(config["storage"]["tags"], qid, stream, feature, tag), 
                                                out_path=f"video_tags/{stream}/{feature}/{tag}",
                                                mime_type="application/json"))
        try:
            client.upload_files(qwt, qlib, file_jobs)
        except HTTPError as e:
            return Response(response=json.dumps({'error': str(e), 'message': 'Please verify you\'re authorization token has write access'}), status=403, mimetype='application/json')
        
        return Response(response=json.dumps({'message': 'Succesfully uploaded tag files. Please finalize the write token.', 'write token': qwt}), status=200, mimetype='application/json')
    
    # JobStatus represents the status of a job returned by the /status endpoint
    @dataclass
    class JobStatus():
        status: Literal["Starting", 
                        "Fetching parts",
                        "Waiting to be assigned GPU", 
                        "Completed", 
                        "Failed", 
                        "Stopped"]
        # time running (in seconds)
        time_running: float
        tag_job_id: Optional[str]=None
        error: Optional[str]=None

    def _get_job_status(job: Job) -> JobStatus:
        if job.time_ended is None:
            time_running = time.time() - job.time_started
        else:
            time_running = job.time_ended - job.time_started
        return JobStatus(status=job.status, time_running=time_running, tag_job_id=job.tag_job_id, error=job.error)
    
    # get status of all jobs for a qid
    @app.route('/<qid>/status', methods=['GET'])
    def status(qid: str) -> Response:
        _, error_response = _get_client(request, qid, config["fabric"]["config_url"])
        if error_response:
            return error_response
        with lock:
            jobs = set(active_jobs[qid].keys()) | set(inactive_jobs[qid].keys())
            if len(jobs) == 0:
                return Response(response=json.dumps({'error': f"No jobs started for {qid}"}), status=404, mimetype='application/json')
            res = defaultdict(dict)
            for job in jobs:
                stream, feature = job
                if job in active_jobs[qid]:
                    res[stream][feature] = _get_job_status(active_jobs[qid][job])
                else:
                    res[stream][feature] = _get_job_status(inactive_jobs[qid][job])
        for stream in list(res.keys()):
            for feature in list(res[stream].keys()):
                res[stream][feature] = asdict(res[stream][feature])
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
    
    @app.route('/<qid>/stop/<feature>', methods=['POST'])
    def stop(qid: str, feature: str) -> Response:
        _, error_response = _get_client(request, qid, config["fabric"]["config_url"])
        if error_response: 
            return error_response
        with lock:
            job_keys = active_jobs[qid].keys()
            feature_job_keys = [job_key for job_key in job_keys if job_key[1] == feature]
            if len(feature_job_keys) == 0:
                return Response(response=json.dumps({'error': f"No job running for {feature} on {qid}"}), status=404, mimetype='application/json')
            jobs = [active_jobs[qid][job_key] for job_key in feature_job_keys]
            for job in jobs:
                job.stop_event.set()
        return Response(response=json.dumps({'message': f"Stopping {feature} on {qid}. Check with /status for completion."}), status=200, mimetype='application/json')
    
    def _list_services() -> List[str]:
        with PodmanClient() as podman_client:
            images = [image.tags[0] for image in podman_client.images.list() if image.tags]
        res = []
        for service in config['services']:
            if config['services'][service]['image'] in images:
                res.append(service)
            else:
                logger.error(f"Image {config['services'][service]['image']} not found")
        return res
    
    # Not thread safe
    # Should be called with lock
    def _set_stop_status(job: Job, status: str, error: Optional[str]=None) -> None:
        qid, stream, feature = job.qid, job.run_config.stream, job.feature
        job = active_jobs[qid][(stream, feature)]
        job.status = status
        job.error = error
        job.time_ended = time.time()
        inactive_jobs[qid][(stream, feature)] = job
        del active_jobs[qid][(stream, feature)]
        
    def _get_client(request: Request, qid: str, config_url: str) -> Tuple[ElvClient, Optional[Response]]:
        auth = _get_authorization(request)
        if not auth:
            return None, Response(response=json.dumps({'error': 'No authorization provided'}), status=400, mimetype='application/json')
        client = ElvClient.from_configuration_url(config_url=config_url, static_token=auth)
        if not _authenticate(client, qid):
            return None, Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
        return client, None

    def _get_authorization(req: Request) -> Optional[str]:
        auth = req.headers.get('Authorization', None)
        if auth:
            return auth
        return req.args.get('authorization', None)
    
    # Basic authentication against the object
    def _authenticate(client: ElvClient, qid: str) -> bool:
        try:
            client.content_object(qid)
        except HTTPError:
            return False
        return True
    
    def _get_invalid_features(features: Iterable[str]) -> List[str]:
        services = _list_services()
        return [feature for feature in features if feature not in services]
    
    def _shutdown() -> None:
        logger.warning("Shutting down")
        to_stop = []
        with lock:
            for qid in active_jobs:
                for job in active_jobs[qid].values():
                    to_stop.append(job)
                    job.stop_event.set()
            logger.info(f"Stopping {len(to_stop)} jobs")
        while True:
            exit = True
            for job in to_stop:
                if not _is_job_stopped(job):
                    exit = False
            if exit:
                # quit loop and finish
                break
            time.sleep(1)
        logger.info("All jobs stopped")
        # Uses os._exit to avoid calling atexit functions
        os._exit(0)
        
    # not thread safe, call with lock
    def _is_job_stopped(job: Job) -> bool:
        return job.status == "Stopped" or job.status == "Failed"
    
    def _startup():
        threading.Thread(target=_job_watcher, daemon=True).start()
        threading.Thread(target=_job_starter, daemon=True).start()
    
    # handle shutdown signals
    signal.signal(signal.SIGINT, lambda sig, frame: _shutdown())
    signal.signal(signal.SIGTERM, lambda sig, frame: _shutdown())

    # in case of a different cause for shutdown
    atexit.register(_shutdown)
            
    _startup()
    CORS(app)
    return app

def main(): 
    app = get_flask_app()
    app.run(port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8086)
    args = parser.parse_args()
    main()