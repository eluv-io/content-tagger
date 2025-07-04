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
from marshmallow import ValidationError, fields, Schema
import tempfile

from common_ml.types import Data
from common_ml.utils.metrics import timeit

from config import config
from src.fabric.utils import parse_qhit
from src.fabric.agg import format_video_tags, format_asset_tags
from src.fabric.video import download_stream, StreamNotFoundError
from src.fabric.assets import fetch_assets, AssetsNotFoundException

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
    qhit: str
    feature: str
    run_config: RunConfig
    stop_event: threading.Event
    media_files: List[str]
    replace: bool
    time_started: float
    failed: List[str]
    allowed_gpus: List[str]
    time_ended: Optional[float]=None
    # tag_job_id is the job id returned by the manager, will be None until the tagging starts (status is "Running")
    tag_job_id: Optional[str]=None
    error: Optional[str]=None
    
def get_flask_app():
    app = Flask(__name__)
    # manages the gpu and inference jobs
    manager = ResourceManager()
    
    # queue for jobs waiting to be assigned a GPU
    gpu_queue = Queue()

    # queue for jobs waiting to be submitted that don't require a GPU
    cpu_queue = Queue()

    # locks active, and inactive jobs dictionaries
    lock = threading.Lock()
    # maps qhit -> (feature, stream) -> job
    active_jobs = defaultdict(dict)
    inactive_jobs = defaultdict(dict)
    
    # need for some filesytem operations
    filesystem_lock = threading.Lock()

    # make sure no two streams are being downloaded at the same time. 
    # maps (qhit, stream) -> lock
    download_lock = defaultdict(threading.Lock)
    
    shutdown_signal = threading.Event()

    @app.route('/list', methods=['GET'])
    def list_services() -> Response:
        res = _list_services()    
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
        
    @app.route('/<qhit>/upload_tags', methods=['POST'])
    def upload(qhit: str):
        uploaded_files = request.files.getlist('file')
        if len(uploaded_files) == 0:
            return Response(response=json.dumps({'error': 'No files in request'}), status=400, mimetype='application/json')
        
        _, error_response = _get_client(request, qhit, config["fabric"]["config_url"])
        if error_response:
            return error_response

        to_upload = []
        for file in uploaded_files:
            try:
                filedata = json.load(file.stream)
            except json.JSONDecodeError:
                return Response(response=json.dumps({'error': 'Invalid JSON file'}), status=400, mimetype='application/json')
            to_upload.append((file.filename, filedata))

        os.makedirs(os.path.join(config["storage"]["tags"], qhit, 'external_tags'), exist_ok=True)

        for fname, fdata in to_upload:
            if os.path.exists(os.path.join(config["storage"]["tags"], qhit, 'external_tags', fname)):
                logger.warning(f"File {fname} already exists, overwriting")
            with open(os.path.join(config["storage"]["tags"], qhit, 'external_tags', fname), 'w') as f:
                json.dump(fdata, f)

        return Response(response=json.dumps({'message': 'Successfully uploaded tags'}), status=200, mimetype='application/json')
    
    # TagArgs represents the request body for the /tag endpoint
    @dataclass
    class TagArgs(Data):
        # maps feature name to RunConfig
        features: Dict[str, RunConfig]
        # start_time in milliseconds (defaults to 0)
        start_time: Optional[int]=None
        # end_time in milliseconds (defaults to entire content)
        end_time: Optional[int]=None
        # replace tag files if they already exist
        replace: bool=False

        @staticmethod
        def from_dict(data: dict) -> 'TagArgs':
            features = {feature: RunConfig(**cfg) for feature, cfg in data['features'].items()}
            return TagArgs(features=features, start_time=data.get('start_time', None), end_time=data.get('end_time', None), replace=data.get('replace', False))

    @app.route('/<qhit>/tag', methods=['POST'])
    def tag(qhit: str) -> Response:
        try:
            args = TagArgs.from_dict(request.json)
        except (KeyError, TypeError) as e:
            return Response(response=json.dumps({'error': f"Invalid input: {str(e)}"}), status=400, mimetype='application/json')
        
        _, error_response = _get_client(request, qhit, config["fabric"]["parts_url"])
        if error_response:
            return error_response
        
        invalid_services = _get_invalid_features(args.features.keys())
        if invalid_services:
            return Response(response=json.dumps({'error': f"Services {invalid_services} not found"}), status=404, mimetype='application/json')
            
        with lock:
            if shutdown_signal.is_set():
                return Response(response=json.dumps({'error': 'Server is shutting down'}), status=503, mimetype='application/json')
            for feature, run_config in args.features.items():
                if run_config.stream is None:
                    # if stream name is not provided, we pick stream based on whether the model is audio/video based
                    run_config.stream = config["services"][feature]["type"]
                if (run_config.stream, feature) in active_jobs[qhit]:
                    return Response(response=json.dumps({'error': f"{feature} tagging is already in progress for {qhit} on {run_config.stream}"}), status=400, mimetype='application/json')
            for feature, run_config in args.features.items():
                # get the subset of GPUs that the model can run on, default to all of them
                allowed_gpus = config["services"][feature].get("allowed_gpus", list(range(manager.num_devices)))
                logger.debug(f"Starting {feature} on {qhit}: {run_config.stream}")
                job = Job(qhit=qhit, run_config=run_config, feature=feature, media_files=[], failed=[], replace=args.replace, status="Starting", stop_event=threading.Event(), time_started=time.time(), allowed_gpus=allowed_gpus)
                active_jobs[qhit][(run_config.stream, feature)] = job
                threading.Thread(target=_video_tag, args=(job, _get_authorization(request), args.start_time, args.end_time)).start()
        return Response(response=json.dumps({'message': f'Tagging started on {qhit}'}), status=200, mimetype='application/json')
    
    def _video_tag(job: Job, authorization: str, start_time: Optional[int], end_time: Optional[int]) -> None:
        part_download_client = ElvClient.from_configuration_url(config_url=config["fabric"]["parts_url"], static_token=authorization)
        media_files, failed = _download_content(job, part_download_client, start_time=start_time, end_time=end_time)
        job.media_files = media_files
        job.failed = failed
        _submit_tag_job(job, ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=authorization))

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
            cpu_queue.put(job)

    def _filter_tagged_files(media_files: List[str], client: ElvClient, qhit: str, stream: str, feature: str) -> List[str]:
        """
        Args:
            media_files (List[str]): list of media files to filter
            qhit (str): content object, hash, or write token that files belong to
            stream (str): stream name
            feature (str): model name

        Returns:
            List[str]: list of media files that have not been tagged, filtered subset of media_files
        """
        content_args = parse_qhit(qhit)
        qlib = client.content_object_library_id(**content_args)
        try:
            if stream == "image":
                tag_files = client.list_files(qlib, path=f"image_tags/{feature}", **content_args)
            else:
                tag_files = client.list_files(qlib, path=f"video_tags/{stream}/{feature}", **content_args)
        except HTTPError:
            # if the folder doesn't exist, then no files have been tagged
            return media_files[:]
        tagged = set(_source_from_tag_file(tag) for tag in tag_files)
        untagged = []
        for media_file in media_files:
            filename = os.path.basename(media_file)
            if filename not in tagged:
                untagged.append(media_file)
        return untagged
    
    def _source_from_tag_file(tagged_file: str) -> str:
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

     # TagArgs represents the request body for the /tag endpoint
    @dataclass
    class ImageTagArgs(Data):
        # maps feature name to RunConfig
        features: Dict[str, RunConfig]

        # asset file paths to tag relative to the content object e.g. /assets/image.jpg, if empty then we will look in /meta/assets and tag all the image assets located there. 
        assets: Optional[List[str]]

        # replace tag files if they already exist
        replace: bool=False

        @staticmethod
        def from_dict(data: dict) -> 'ImageTagArgs':
            features = {feature: RunConfig(stream='image', **cfg) for feature, cfg in data['features'].items()}
            return ImageTagArgs(features=features, assets=data.get('assets', None), replace=data.get('replace', False))
        
    @app.route('/<qhit>/image_tag', methods=['POST'])
    def image_tag(qhit: str) -> Response:
        try:
            args = ImageTagArgs.from_dict(request.json)
        except TypeError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        
        _, error_response = _get_client(request, qhit, config["fabric"]["config_url"])
        if error_response:
            return error_response
        
        invalid_services = _get_invalid_features(args.features.keys())
        if invalid_services:
            return Response(response=json.dumps({'error': f"Services {invalid_services} not found"}), status=404, mimetype='application/json')

        for feature, run_config in args.features.items():
            if not config["services"][feature].get("frame_level", False):
                return Response(response=json.dumps({'error': f"Image tagging for {feature} is not supported"}), status=400, mimetype='application/json')
            
        with lock:
            if shutdown_signal.is_set():
                return Response(response=json.dumps({'error': 'Server is shutting down'}), status=503, mimetype='application/json')
            for feature in args.features.keys():
                if active_jobs[qhit].get(('image', feature), None):
                    return Response(response=json.dumps({'error': f"Image tagging for at least one of the requested features, {feature}, is already in progress for {qhit}"}), status=400, mimetype='application/json')
            for feature, run_config in args.features.items():
                # get the subset of GPUs that the model can run on, default to all of them
                allowed_gpus = config["services"][feature].get("allowed_gpus", list(range(manager.num_devices)))
                job = Job(qhit=qhit, feature=feature, run_config=run_config, media_files=[], failed=[], replace=args.replace, allowed_gpus=allowed_gpus, status="Starting", stop_event=threading.Event(), time_started=time.time())
                active_jobs[qhit][('image', feature)] = job
                threading.Thread(target=_image_tag, args=(job, _get_authorization(request), args.assets)).start()
        return Response(response=json.dumps({'message': f'Image asset tagging started on {qhit}'}), status=200, mimetype='application/json')
    
    def _image_tag(job: Job, authorization: str, assets: Optional[List[str]]) -> None:
        elv_client = ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=authorization)
        images, failed = _download_content(job, elv_client, assets=assets)
        deduped = list(set(images))
        if len(deduped) > 0:
            logger.warning(f"Found {len(images) - len(deduped)} duplicate images.")
        job.media_files = deduped
        job.failed = failed
        _submit_tag_job(job, elv_client)
    
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

    def _job_starter(wait_for_gpu: bool, tag_queue: Queue) -> None:
        """
        Args:
            wait_for_gpu (bool): if True, then the job starter will wait for a GPU to be available before starting the job
            tag_queue (Queue): queue for jobs waiting to be submitted
        """
        while True:
            # blocks until a job is available
            job = tag_queue.get()
            if _check_exit(job):
                continue
            
            if 0 < len(job.allowed_gpus) < manager.num_devices:
                # This means the job can only run on a subset of GPUs
                # The sleep prevents the situation where this type of job is added back to queue over and over while it waits for a GPU.
                time.sleep(config["devices"]["wait_for_gpu_sleep"])

            if wait_for_gpu:
                stopped = False
                while not manager.await_gpu(timeout=config["devices"]["wait_for_gpu_sleep"]):
                    # check if the job has been stopped, if not then we go back to waiting for a GPU
                    if _check_exit(job):
                        stopped = True
                        break
                if stopped:
                    continue
            try:
                job_id = manager.run(job.feature, job.run_config.model, job.media_files, job.allowed_gpus)
                with lock:
                    if _is_job_stopped(job):
                        # if the job has been stopped while the container was starting
                        manager.stop(job.tag_job_id)
                        continue
                    job.tag_job_id = job_id
                    job.status = "Tagging content"
                logger.success(f"Started running {job.feature} on {job.qhit}")
            except NoGPUAvailable:
                # This error can happen if the model can only run on a subset of GPUs. 
                job.error = "Tried to assign GPU but no suitable one was found. The job was placed back on the queue."
                tag_queue.put(job)
            except Exception as e:
                # handle the unexpected
                with lock:
                    _set_stop_status(job, "Failed", str(e))
                    
    def _job_watcher() -> None:
        while True:
            for qhit in active_jobs:
                for (stream, feature), job in list(active_jobs[qhit].items()):
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
                        logger.success(f"Finished running {job.feature} on {job.qhit}")
                        with filesystem_lock:
                            # move outputted tags to their correct place
                            # lock in case of race condition with status or finalize calls
                            _move_files(job, status.tags)
                    job.status = status.status
                    if status.status == "Failed":
                        job.error = "An error occurred while running model container"
                    job.time_ended = time.time()
                    with lock:
                        # move job to inactive_jobs
                        inactive_jobs[qhit][(stream, feature)] = job
                        del active_jobs[qhit][(stream, feature)]
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

    def _move_files(job: Job, tags: List[str]) -> None:
        if len(tags) == 0:
            return
        tag_dir = os.path.dirname(tags[0])
        qhit, stream, feature = job.qhit, job.run_config.stream, job.feature
        tags_path = os.path.join(config["storage"]["tags"], qhit, stream, feature)
        os.makedirs(tags_path, exist_ok=True)
        for tag in tags:
            shutil.move(tag, os.path.join(tags_path, os.path.basename(tag)))
        shutil.rmtree(tag_dir, ignore_errors=True)
    
    @dataclass
    class FinalizeArgs(Data):
        write_token: str
        replace: bool=False
        force: bool=False
        # if live is set, then we don't finalize the file job
        leave_open: bool=False
        authorization: Optional[str]=None

        @staticmethod
        def from_dict(data: dict) -> 'FinalizeArgs':
            class FinalizeSchema(Schema):
                write_token = fields.Str(required=True)
                replace = fields.Bool(required=False, missing=False)
                force = fields.Bool(required=False, missing=False)
                leave_open = fields.Bool(required=False, missing=False)
                authorization = fields.Str(required=False, missing=None)
            return FinalizeArgs(**FinalizeSchema().load(data))

    @app.route('/<qhit>/finalize', methods=['POST'])
    def finalize(qhit: str) -> Response:
        return _finalize_internal(qhit, True)

    @app.route('/<qhit>/aggregate', methods=['POST'])
    def aggregate(qhit: str) -> Response:
        return _finalize_internal(qhit, False)
    
    def _finalize_internal(qhit: str, upload_local_tags = True) -> Response:
        try:
            args = FinalizeArgs.from_dict(request.args)
        except (TypeError, ValidationError) as e:
            return Response(response=json.dumps({'message': 'invalid request', 'error': str(e)}), status=400, mimetype='application/json')
        qwt = args.write_token
        client, error_response = _get_client(request, qwt, config["fabric"]["config_url"])
        if error_response:
            return error_response
        # TODO: if write token doesn't exist it will give 404, but we return 403 unauthorized which is misleading.
        if not _authenticate(client, qhit):
            # make sure that the auth token has access to the content object where the tags are from
            # TODO: we may need to check more permissions to make sure the user should be able to read the tags. 
            return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
        content_args = parse_qhit(qwt)
        qlib = client.content_object_library_id(**content_args)

        file_jobs = []
        if upload_local_tags:
            with lock:
                jobs_running = len(active_jobs[qhit].values())
            if jobs_running > 0 and not args.force:
                return Response(response=json.dumps({'error': 'Some jobs are still running. Use `force=true` to finalize anyway.'}), status=400, mimetype='application/json')

            if not os.path.exists(os.path.join(config["storage"]["tags"], qhit)):
                return Response(response=json.dumps({'error': 'No tags found for this content object'}), status=404, mimetype='application/json')

            for stream in os.listdir(os.path.join(config["storage"]["tags"], qhit)):
                if stream == "external_tags":
                    continue
                for feature in os.listdir(os.path.join(config["storage"]["tags"], qhit, stream)):
                    tagged_media_files = []
                    for tag in os.listdir(os.path.join(config["storage"]["tags"], qhit, stream, feature)):
                        tagfile = os.path.join(config["storage"]["tags"], qhit, stream, feature, tag)
                        tagged_media_files.append(_source_from_tag_file(tagfile))
                    tagged_media_files = list(set(tagged_media_files))
                    num_files = len(tagged_media_files)
                    if not args.replace:
                        with timeit(f"Filtering tagged files for {qhit}, {feature}, {stream}"):
                            tagged_media_files = _filter_tagged_files(tagged_media_files, client, qhit, stream, feature)
                    logger.debug(f"Upload status for {qhit}: {feature} on {stream}\nTotal media files: {num_files}, Media files to upload: {len(tagged_media_files)}, Media files already uploaded: {num_files - len(tagged_media_files)}")
                    if not tagged_media_files:
                        continue
                    if stream == "image":
                        for source in tagged_media_files:
                            tagfile = source + "_imagetags.json"
                            file_jobs.append(ElvClient.FileJob(local_path=tagfile,
                                                    out_path=f"image_tags/{feature}/{os.path.basename(tagfile)}",
                                                    mime_type="application/json"))
                    else:
                        for source in tagged_media_files:
                            tagfile = source + "_tags.json"
                            file_jobs.append(ElvClient.FileJob(local_path=tagfile,
                                                    out_path=f"video_tags/{stream}/{feature}/{os.path.basename(tagfile)}",
                                                    mime_type="application/json"))
                            if os.path.exists(source + "_frametags.json"):
                                file_jobs.append(ElvClient.FileJob(local_path=source + "_frametags.json",
                                                    out_path=f"video_tags/{stream}/{feature}/{os.path.basename(source)}_frametags.json",
                                                    mime_type="application/json"))
                                
            external_tags_path = os.path.join(config["storage"]["tags"], qhit, "external_tags")
            if os.path.exists(external_tags_path):
                local_source_tags = os.listdir(external_tags_path)
                try:
                    remote_source_tags = client.list_files(qlib, path="video_tags/source_tags/external", **content_args)
                except HTTPError:
                    logger.debug(f"No source tags found for {qwt}")
                    remote_source_tags = []
                
                for local_source in local_source_tags:
                    tagfile = os.path.join(config["storage"]["tags"], qhit, "external_tags", local_source)
                    file_jobs.append(ElvClient.FileJob(local_path=tagfile,
                                                        out_path=f"video_tags/source_tags/external/{local_source}",
                                                        mime_type="application/json"))
                    # TODO: we do need a way to make it so we can do replace=true for external tags but not on the rest. if we can improve the efficiency of this step we could just do two passes and 
                    # Let the user specify which features they want to finalize, and they could do two steps. For now, we will default to always overwriting the external tags.
                    if local_source in remote_source_tags:
                        logger.warning(f"External tag file {local_source} already exists, overwriting")

        if len(file_jobs) > 0:
            try:
                logger.debug(f"Uploading {len(file_jobs)} tag files")
                with timeit("Uploading tag files"):
                    client.upload_files(library_id=qlib, file_jobs=file_jobs, finalize=False, **content_args)
            except HTTPError as e:
                return Response(json.dumps({'error': str(e), 'message': 'Please verify your authorization token has write access and the write token has not already been committed. This error can also arise if the write token has already been used to finalize tags.'}), status=403, mimetype='application/json')
            except ValueError as e:
                return Response(response=json.dumps({'error': str(e), 'message': 'Please verify the provided write token has not already been used to finalize tags.'}), status=400, mimetype='application/json')
        # if no file jobs, then we just do the aggregation
        
        tmpdir = tempfile.TemporaryDirectory(dir=config["storage"]["tmp"])
        with filesystem_lock:
            shutil.copytree(os.path.join(config["storage"]["tags"], qhit), tmpdir.name, dirs_exist_ok=True)
            if os.path.exists(os.path.join(tmpdir.name, 'external_tags')):
                shutil.rmtree(os.path.join(tmpdir.name, 'external_tags'))

        try:
            with timeit("Aggregating video tags"):
                format_video_tags(client, qwt, config["agg"]["interval"], tmpdir.name)
            with timeit("Aggregating asset tags"):
                format_asset_tags(client, qwt, tmpdir.name)
        except HTTPError as e:
            message = (
                "Please verify your authorization token has write access and the write token has not already been committed."
                "This error can also arise if the write token has already been used to finalize tags."
            )
            return Response(response=json.dumps({'error': str(e), 'message': message}), status=403, mimetype='application/json')
        finally:
            tmpdir.cleanup()
        
        if not args.leave_open:
            client.finalize_files(qwt, qlib)

        client.set_commit_message(qwt, "Uploaded ML Tags", qlib)

        return Response(response=json.dumps({'message': 'Succesfully uploaded tag files. Please finalize the write token.', 'write token': qwt}), status=200, mimetype='application/json')

    # JobStatus represents the status of a job returned by the /status endpoint
    @dataclass
    class JobStatus():
        status: Literal["Starting", 
                        "Fetching content",
                        "Waiting to be assigned GPU", 
                        "Tagging content",
                        "Completed", 
                        "Failed", 
                        "Stopped"]
        # time running (in seconds)
        time_running: float
        tagging_progress: str
        tag_job_id: Optional[str]=None
        error: Optional[str]=None
        failed: List[str]=field(default_factory=list)

    def _get_job_status(job: Job) -> JobStatus:
        if job.time_ended is None:
            time_running = time.time() - job.time_started
        else:
            time_running = job.time_ended - job.time_started
        
        if job.status == "Tagging content":
            with filesystem_lock:
                # read how many files have been tagged
                tag_dir = os.path.join(config["storage"]["tmp"], job.feature, job.tag_job_id)
                if not os.path.exists(tag_dir):
                    tagged_files = 0
                else:
                    tagged_files = len(os.listdir(tag_dir))
            if job.run_config.stream != "image" and config["services"][job.feature].get("frame_level", False):
                # for frame level tagging on video we have two tag files per part, so we need to divide by two.
                tagged_files = tagged_files // 2
            to_tag = len(job.media_files)
            progress = f"{tagged_files}/{to_tag}"
        elif job.status == "Completed":
            tagged_files = len(job.media_files)
            progress = f"{tagged_files}/{tagged_files}"
        else:
            progress = ""
            
        return JobStatus(status=job.status, time_running=time_running, tagging_progress=progress, tag_job_id=job.tag_job_id, error=job.error, failed=job.failed)
    
    @app.route('/<qhit>/status', methods=['GET'])
    def status(qhit: str) -> Response:
        """Get the status of all tag jobs for a given qhit"""
        _, error_response = _get_client(request, qhit, config["fabric"]["config_url"])
        if error_response:
            return error_response
        with lock:
            jobs = set(active_jobs[qhit].keys()) | set(inactive_jobs[qhit].keys())
            if len(jobs) == 0:
                return Response(response=json.dumps({'error': f"No jobs started for {qhit}"}), status=404, mimetype='application/json')
            res = defaultdict(dict)
            for job in jobs:
                stream, feature = job
                if job in active_jobs[qhit]:
                    res[stream][feature] = _get_job_status(active_jobs[qhit][job])
                else:
                    res[stream][feature] = _get_job_status(inactive_jobs[qhit][job])
        for stream in list(res.keys()):
            for feature in list(res[stream].keys()):
                res[stream][feature] = asdict(res[stream][feature])
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
    
    @app.route('/<qhit>/stop/<feature>', methods=['POST'])
    def stop(qhit: str, feature: str) -> Response:
        _, error_response = _get_client(request, qhit, config["fabric"]["config_url"])
        if error_response: 
            return error_response
        with lock:
            job_keys = active_jobs[qhit].keys()
            feature_job_keys = [job_key for job_key in job_keys if job_key[1] == feature]
            if len(feature_job_keys) == 0:
                return Response(response=json.dumps({'error': f"No job running for {feature} on {qhit}"}), status=404, mimetype='application/json')
            jobs = [active_jobs[qhit][job_key] for job_key in feature_job_keys]
            for job in jobs:
                job.stop_event.set()
        return Response(response=json.dumps({'message': f"Stopping {feature} on {qhit}. Check with /status for completion."}), status=200, mimetype='application/json')
    
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
        qhit, stream, feature = job.qhit, job.run_config.stream, job.feature
        job = active_jobs[qhit][(stream, feature)]
        job.status = status
        job.error = error
        job.time_ended = time.time()
        inactive_jobs[qhit][(stream, feature)] = job
        del active_jobs[qhit][(stream, feature)]
        
    def _get_client(request: Request, qhit: str, config_url: str) -> Tuple[ElvClient, Optional[Response]]:
        auth = _get_authorization(request)
        if not auth:
            return None, Response(response=json.dumps({'error': 'No authorization provided'}), status=400, mimetype='application/json')
        client = ElvClient.from_configuration_url(config_url=config_url, static_token=auth)
        if not _authenticate(client, qhit):
            return None, Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
        return client, None

    def _get_authorization(req: Request) -> Optional[str]:
        auth = req.headers.get('Authorization', None)
        if auth:
            return auth
        return req.args.get('authorization', None)
    
    # Basic authentication against the object
    def _authenticate(client: ElvClient, qhit: str) -> bool:
        try:
            client.content_object(**parse_qhit(qhit))
        except HTTPError as e:
            logger.error(e)
            return False
        return True
    
    def _get_invalid_features(features: Iterable[str]) -> List[str]:
        services = _list_services()
        return [feature for feature in features if feature not in services]
    
    def _shutdown() -> None:
        logger.warning("Shutting down")
        to_stop = []
        shutdown_signal.set()
        with lock:
            for qhit in active_jobs:
                for job in active_jobs[qhit].values():
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
        threading.Thread(target=_job_starter, args=(True, gpu_queue), daemon=True).start()
        threading.Thread(target=_job_starter, args=(False, cpu_queue), daemon=True).start()
    
    # handle shutdown signals
    signal.signal(signal.SIGINT, lambda sig, frame: _shutdown())
    signal.signal(signal.SIGTERM, lambda sig, frame: _shutdown())

    # in case of a different cause for shutdown other than a termination signal
    atexit.register(_shutdown)
            
    _startup()
    CORS(app)
    return app

def main():
    app = get_flask_app()
    app.run(port=args.port, host=args.host)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--host', type=str, default="127.0.0.1")
    args = parser.parse_args()
    main()
