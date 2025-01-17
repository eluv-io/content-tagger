import argparse
from typing import List, Optional, Literal, Dict, Iterable
from flask import Flask, request, Response, Request
from flask_cors import CORS
from podman import PodmanClient
import json
from loguru import logger
from dataclasses import dataclass, asdict, field
import os
from elv_client_py import ElvClient
from collections import defaultdict, deque
import threading
from requests.exceptions import HTTPError
import traceback
import time
import shutil
import signal
import atexit
from common_ml.types import Data

from config import config
from src.fetch import fetch_stream, StreamNotFoundError, fetch_assets, AssetsNotFoundException
from src.manager import ResourceManager, NoGPUAvailable

@dataclass
class Job:
    status: Literal["Starting", 
                    "Fetching content",
                    "Waiting to be assigned GPU", 
                    "Completed", 
                    "Failed", 
                    "Stopped"]
    stop_event: threading.Event
    time_started: float
    time_ended: Optional[float]=None
    # tag_job_id is the job id returned by the manager, will be None until the tagging starts (status is "Running")
    tag_job_id: Optional[str]=None
    error: Optional[str]=None
    message: Optional[str]=None

def get_flask_app():
    app = Flask(__name__)
    # manages the gpu and inference jobs, is thread safe
    manager = ResourceManager()

    # locks job_queue, active, and inactive jobs dictionaries
    lock = threading.Lock()
    job_queue = deque([])
    # maps qid -> (feature, stream) -> job
    # if job is waiting to be assigned a GPU it will still go in active_jobs
    active_jobs = defaultdict(dict)
    inactive_jobs = defaultdict(dict)

    # make sure no two streams are being downloaded at the same time. 
    # maps (qid, stream) -> lock
    download_lock = defaultdict(threading.Lock)

    @app.route('/list', methods=['GET'])
    def list_services() -> Response:
        res = _list_services()    
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
    
    # RunConfig gives model level tagging params
    @dataclass
    class RunConfig():
        # model config, used to overwrite the model level config
        model: dict=field(default_factory=dict) 
        # stream name to run the model on, None to use the default stream. "image" is a special case which will tag image assets
        stream: Optional[str]=None 
    
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
        data = request.json
        try:
            args = TagArgs.from_dict(data)
        except TypeError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        auth = _get_authorization(request)
        if not auth:
            return Response(response=json.dumps({'error': 'No authorization provided'}), status=400, mimetype='application/json')
        # elv_client should point to a url which offers the parts_download service
        elv_client = ElvClient.from_configuration_url(config_url=config["fabric"]["parts_url"], static_token=auth)
        if not _authenticate(elv_client, qid):
            return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
        
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
                active_jobs[qid][(run_config.stream, feature)] = Job(status="Starting", stop_event=threading.Event(), time_started=time.time())
            threading.Thread(target=_video_tag, args=(args.features, qid, elv_client, args.start_time, args.end_time)).start()
        return Response(response=json.dumps({'message': f'Tagging started on {qid}'}), status=200, mimetype='application/json')
    
    def _video_tag(feature: str, run_config: RunConfig, qid: str, elv_client: ElvClient, start_time: Optional[int], end_time: Optional[int]) -> None:
        feature_to_media_files = _download_content(features, qid, elv_client, start_time=start_time, end_time=end_time)
        _tag(features, feature_to_media_files, qid)

     # TagArgs represents the request body for the /tag endpoint
    @dataclass
    class ImageTagArgs(Data):
        # maps feature name to RunConfig
        features: Dict[str, RunConfig]

        # asset file paths to tag relative to the content object e.g. /assets/image.jpg, if empty then we will look in /meta/assets and tag all the image assets located there. 
        assets: Optional[List[str]]

        # if true, will overwrite existing tags. 
        replace: bool

        @staticmethod
        def from_dict(data: dict) -> 'ImageTagArgs':
            features = {feature: RunConfig(stream='image', **cfg) for feature, cfg in data['features'].items()}
            return ImageTagArgs(features=features, assets=data.get('assets', None), replace=data.get('replace', False))
        
    @app.route('/<qid>/image_tag', methods=['POST'])
    def image_tag(qid: str) -> Response:
        data = request.json
        try:
            args = ImageTagArgs.from_dict(data)
        except TypeError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        auth = _get_authorization(request)
        if not auth:
            return Response(response=json.dumps({'error': 'No authorization provided'}), status=400, mimetype='application/json')
        # elv_client should point to a url which offers the parts_download service
        elv_client = ElvClient.from_configuration_url(config_url=config["fabric"]["parts_url"], static_token=auth)
        if not _authenticate(elv_client, qid):
            return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
        
        invalid_services = _get_invalid_features(args.features.keys())
        if invalid_services:
            return Response(response=json.dumps({'error': f"Services {invalid_services} not found"}), status=404, mimetype='application/json')
            
        with lock:
            for feature in args.features.keys():
                if active_jobs[qid].get(('image', feature), None):
                    return Response(response=json.dumps({'error': f"Image tagging for at least one of the requested features, {feature}, is already in progress for {qid}"}), status=400, mimetype='application/json')
            for feature in args.features.keys():
                active_jobs[qid][('image', feature)] = Job(status="Starting", stop_event=threading.Event(), time_started=time.time())
            threading.Thread(target=_image_tag, args=(args.features, qid, elv_client, args.assets, args.replace)).start()
        return Response(response=json.dumps({'message': f'Image asset tagging started on {qid}'}), status=200, mimetype='application/json')
    
    def _image_tag(features: Dict[str, RunConfig], qid: str, elv_client: ElvClient, assets: Optional[List[str]], replace: bool) -> None:
        feature_to_images = _download_content(features, qid, elv_client, assets=assets)
        _tag(features, feature_to_images, qid)
    
    # Download content and tag
    def _download_content(features: Dict[str, RunConfig], qid: str, elv_client: ElvClient, **kwargs) -> Dict[str, List[str]]:
        feature_to_media_files = {}
        for feature, cfg in features.items():
            stream = cfg.stream
            with lock:
                job = active_jobs[qid][(stream, feature)]
            if job.stop_event.is_set():
                with lock:
                    _set_stop_status(qid, stream, feature)
                continue
            # parts storage path
            if stream == "image":
                save_path = os.path.join(config["storage"]["images"], qid, stream)
            else:
                save_path = os.path.join(config["storage"]["parts"], qid, stream)
            try:
                with download_lock[(qid, stream)]:
                    job.status = "Fetching content"

                    # if fetching finished while waiting for lock, this will return immediately
                    if stream == "image":
                        downloaded_files = fetch_assets(qid, save_path, elv_client, **kwargs, exit_event=job.stop_event)
                    else:
                        downloaded_files =  fetch_stream(qid, stream, save_path, elv_client, **kwargs, exit_event=job.stop_event)
                if job.stop_event.is_set():
                    with lock:
                        _set_stop_status(qid, stream, feature)
                    continue
            except (StreamNotFoundError, AssetsNotFoundException):
                with lock:
                    job = active_jobs[qid][(stream, feature)]
                    job.status = "Failed"
                    job.time_ended = time.time()
                    job.error = f"Content for stream {stream}, was not found for {qid}"
                    inactive_jobs[qid][(stream, feature)] = job
                    del active_jobs[qid][(stream, feature)]
                    logger.error(job.error)
                    # continue and skip this feature
                    continue
            except HTTPError as e:
                with lock:
                    job = active_jobs[qid][(stream, feature)]
                    job.status = "Failed"
                    job.error = f"Failed to fetch stream {stream} for {qid}: {str(e)}. Make sure authorization token hasn't expired."
                    job.time_ended = time.time()
                    inactive_jobs[qid][(stream, feature)] = job
                    del active_jobs[qid][(stream, feature)]
                    logger.error(job.error)
                    # continue and skip this feature
                    continue
            feature_to_media_files[feature] = downloaded_files
        return feature_to_media_files        

    # The main tagging function for audio/video/image content. 
    # This manages a set of tag jobs for a single content object.
    # Args:
    #   features: maps feature name to RunConfig
    #   feature_to_media_files: maps feature name to list of media files to tag
    #   qid: content object id
    # Effects:
    #   1. Loops through features to tag and runs tagging container once a GPU is available
    #   2. Waits for tagging to complete
    #   3. Updates the job status appropriately for monitoring progress
    #   4. Moves the outputted tags to the correct location
    # TODO: we need to support an option to avoid tagging when the tags already exist
    def _tag(features: Dict[str, RunConfig], feature_to_media_files: Dict[str, List[str]], qid: str) -> None:
        running_features = []
        for feature in features.keys():
            job = active_jobs[qid][(features[feature].stream, feature)]
            job.status = "Waiting to be assigned GPU"
        for feature, files in feature_to_media_files.items():
            stream = features[feature].stream
            with lock:
                job = active_jobs[qid][(stream, feature)]

            # wait for a GPU to be available to start the tagging process for the job
            while True:
                # check if the job was stopped
                if job.stop_event.is_set():
                    logger.warning(f"Stopping {feature} on {qid}")
                    with lock:
                        _set_stop_status(qid, stream, feature)
                    break
                try:
                    job_id = manager.run(feature, features[feature].model, files)
                except NoGPUAvailable:
                    time.sleep(config["devices"]["wait_for_gpu_sleep"])
                    continue
                except Exception as e:
                    with lock:
                        logger.error(f"Failed to run {feature} on {qid}: {traceback.format_exc()})")
                        job.status = "Failed"
                        job.time_ended = time.time()
                        job.error = str(e)
                        inactive_jobs[qid][(stream, feature)] = job
                        del active_jobs[qid][(stream, feature)]
                        break

                logger.success(f"Started running {feature} on {qid}")
                job.tag_job_id = job_id
                job.status = "Tagging content"

                running_features.append(feature)
                break

        # watch jobs
        while len(running_features) > 0:
            for feature in running_features[:]:
                stream = features[feature].stream
                with lock:
                    job = active_jobs[qid][(stream, feature)]
                if job.stop_event.is_set():
                    _stop_container(job)
                    with lock:
                        _set_stop_status(qid, stream, feature)
                    running_features.remove(feature)
                    continue
                status = manager.status(job.tag_job_id)
                if status.status == "Running":
                    continue
                if status.status == "Completed":
                    # move outputted tags to their correct place
                    _move_files(qid, stream, feature, status.tags)
                job.status = status.status 
                job.time_ended = time.time()
                with lock:
                    inactive_jobs[qid][(stream, feature)] = job
                    del active_jobs[qid][(stream, feature)]
                running_features.remove(feature)
            time.sleep(5)
        logger.success(f"Finished tagging {qid}")

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
        try:
            args = FinalizeArgs(**request.args)
        except TypeError as e:
            return Response(response=json.dumps({'message': 'invalid request', 'error': str(e)}), status=400, mimetype='application/json')
        auth = _get_authorization(request)
        if not auth:
            return Response(response=json.dumps({'error': 'No authorization provided'}), status=400, mimetype='application/json')
        elv_client = ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=auth)
        if not _authenticate(elv_client, qid):
            return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
        qwt = args.write_token
        qlib = elv_client.content_object_library_id(qid)
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
            elv_client.upload_files(qwt, qlib, file_jobs)
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
        message: Optional[str]=None

    def _get_job_status(job: Job) -> JobStatus:
        if job.time_ended is None:
            time_running = time.time() - job.time_started
        else:
            time_running = job.time_ended - job.time_started
        return JobStatus(status=job.status, time_running=time_running, tag_job_id=job.tag_job_id, error=job.error, message=job.message)
    
    # get status of all jobs for a qid
    @app.route('/<qid>/status', methods=['GET'])
    def status(qid: str) -> Response:
        auth = _get_authorization(request)
        if not auth:
            return Response(response=json.dumps({'error': 'No authorization provided'}), status=400, mimetype='application/json')
        client = ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=auth)
        if not _authenticate(client, qid):
            return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')

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
    
    @app.route('/<qid>/stop/<feature>', methods=['DELETE'])
    def stop(qid: str, feature: str) -> Response:
        auth = _get_authorization(request)
        if not auth:
            return Response(response=json.dumps({'error': 'No authorization provided'}), status=400, mimetype='application/json')
        client = ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=auth)
        if not _authenticate(client, qid):
            return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
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
    def _set_stop_status(qid: str, stream: str, feature: str) -> None:
        job = active_jobs[qid][(stream, feature)]
        job.status = "Stopped"
        job.time_ended = time.time()
        job.message = "Job was stopped"
        inactive_jobs[qid][(stream, feature)] = job
        del active_jobs[qid][(stream, feature)]

    def _stop_container(job: Job) -> None:
        manager.stop(job.tag_job_id)
    
    # authentication can be in header or query string
    # returns None if no authorization is found
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
        with lock:
            to_stop = []
            for qid in active_jobs:
                for feature in active_jobs[qid]:
                    job = active_jobs[qid][feature]
                    to_stop.append((qid, feature))
                    job.stop_event.set()
            logger.info(f"Stopping {len(to_stop)} jobs")
        while True:
            exit = True
            with lock:
                for (qid, feature) in to_stop:
                    if feature in active_jobs[qid]:
                        exit = False
            if exit:
                # quit loop and finish
                break
            time.sleep(1)
        logger.info("All jobs stopped")
        os._exit(0)
    
    # graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: _shutdown())
    signal.signal(signal.SIGTERM, lambda sig, frame: _shutdown())

    atexit.register(_shutdown)
            
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