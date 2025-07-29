import argparse
from typing import List, Optional, Literal, Dict
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
from loguru import logger
from dataclasses import dataclass, asdict, field
import os
from elv_client_py import ElvClient
from collections import defaultdict
import threading
from requests.exceptions import HTTPError
import time
import shutil
import signal
import atexit
from marshmallow import ValidationError, fields, Schema
import tempfile
import setproctitle
import sys
from waitress import serve
from common_ml.types import Data
from common_ml.utils.metrics import timeit
import traceback

from config import config, reload_config
from src.fabric.utils import parse_qhit
from src.fabric.agg import format_video_tags, format_asset_tags
from src.tagger.jobs import JobsStore
from src.containers import list_services
from src.api.tagging.handlers import handle_tag
from src.api.errors import BadRequestError, MissingResourceError

from src.tagger.tagger import Tagger
from src.tagger.resource_manager import ResourceManager, NoResourceAvailable

    
## for debugging, keep the last tmpdir (only if set)
last_tmpdir = None

def configure_routes(app: Flask) -> None:
    # Configure the Flask app with the routes defined in this module.

    @app.errorhandler(BadRequestError)
    def handle_bad_request(e):
        tb = traceback.format_exc()
        logger.error(f"Bad request: {e}\n{tb}")
        return jsonify({'message': e.message}), 400

    @app.errorhandler(HTTPError)
    def handle_http_error(e):
        tb = traceback.format_exc()
        logger.error(f"Bad request:\n{tb}")
        status_code = e.response.status_code
        error_resp = json.loads(e.response.text)
        return jsonify({'message': 'Fabric API error', 'error': error_resp}), status_code

    @app.errorhandler(MissingResourceError)
    def handle_missing_resource(e):
        tb = traceback.format_exc()
        logger.error(f"Missing resource: {e}\n{tb}")
        return jsonify({'message': e.message}), 404

    @app.route('/list', methods=['GET'])
    def list_services() -> Response:
        res = list_services()    
        return Response(response=json.dumps(res), status=200, mimetype='application/json')

    @app.route('/<qhit>/tag', methods=['POST'])
    def tag(qhit: str) -> tuple[Response, int]:
        return handle_tag(qhit)

def boot_state(app: Flask) -> None:
    app_state = {}

    app_state["resource_manager"] = ResourceManager()

    app_state["jobs_store"] = JobsStore

    app_state["tagger"] = Tagger

    app.config["state"] = app_state

def get_flask_app():
    app = Flask(__name__)

    

        
    @dataclass
    class UploadArgs(Data):
        aggregate: bool=False
        authorization: Optional[str]=None
        write_token: str=""

        # finalize args, set by default
        leave_open: bool=True
        force: bool=True
        replace: bool=True
        
        @staticmethod
        def from_dict(data: dict) -> 'UploadArgs':
            class UploadArgsSchema(Schema):
                aggregate = fields.Bool(required=False, missing=False)
                write_token = fields.Str(required=False, missing="")
                authorization = fields.Str(required=False, missing=None)
                
            return UploadArgs(**UploadArgsSchema().load(data))
        
    @app.route('/<qhit>/upload_tags', methods=['POST'])
    def upload(qhit: str):
        uploaded_files = request.files.getlist('file')
        if len(uploaded_files) == 0:
            return Response(response=json.dumps({'error': 'No files in request'}), status=400, mimetype='application/json')
        
        try:
            args = UploadArgs.from_dict(request.args)
        except (KeyError, TypeError) as e:
            return Response(response=json.dumps({'error': f"Invalid input: {str(e)}"}), status=400, mimetype='application/json')

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

        with filesystem_lock:
            os.makedirs(os.path.join(config["storage"]["tags"], qhit, 'external_tags'), exist_ok=True)

            for fname, fdata in to_upload:
                if os.path.exists(os.path.join(config["storage"]["tags"], qhit, 'external_tags', fname)):
                    logger.warning(f"File {fname} already exists, overwriting")
                with open(os.path.join(config["storage"]["tags"], qhit, 'external_tags', fname), 'w') as f:
                    json.dump(fdata, f)
                    
        if not args.write_token:
            args.write_token = qhit

        if args.aggregate:
            return _finalize_internal(qhit, args, True)

        return Response(response=json.dumps({'message': 'Successfully uploaded tags'}), status=200, mimetype='application/json')

    @app.route('/<qhit>/tag', methods=['POST'])
    def tag(qhit: str) -> Response:
        handle_tag(qhit)

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
                allowed_cpus = config["services"][feature].get("cpu_slots", [])
                job = Job(qhit=qhit, feature=feature, run_config=run_config, media_files=[], failed=[], replace=args.replace, allowed_gpus=allowed_gpus, allowed_cpus=allowed_cpus, status="Starting", stop_event=threading.Event(), time_started=time.time())
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

    @app.route('/<qhit>/write', methods=['POST'])
    @app.route('/<qhit>/finalize', methods=['POST'])
    def finalize(qhit: str) -> Response:
        try:
            args = FinalizeArgs.from_dict(request.args)
        except (TypeError, ValidationError) as e:
            return Response(response=json.dumps({'message': 'invalid request', 'error': str(e)}), status=400, mimetype='application/json')
        with finalize_lock[args.write_token]:
            return _finalize_internal(qhit, args, True)

    @app.route('/<qhit>/aggregate', methods=['POST'])
    def aggregate(qhit: str) -> Response:
        try:
            args = FinalizeArgs.from_dict(request.args)
        except (TypeError, ValidationError) as e:
            return Response(response=json.dumps({'message': 'invalid request', 'error': str(e)}), status=400, mimetype='application/json')
        with finalize_lock[args.write_token]:
            return _finalize_internal(qhit, args, False)
    
    def _finalize_internal(qhit: str, args: FinalizeArgs, upload_local_tags = True) -> Response:
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
                            if not os.path.exists(tagfile):
                                logger.warning(f"Expected tag file {tagfile} not found, skipping")
                                continue
                            file_jobs.append(ElvClient.FileJob(local_path=tagfile,
                                                    out_path=f"image_tags/{feature}/{os.path.basename(tagfile)}",
                                                    mime_type="application/json"))
                    else:
                        for source in tagged_media_files:
                            tagfile = source + "_tags.json"
                            if not os.path.exists(tagfile):
                                # this should only happen if force=True and frametags get written before video tags
                                logger.warning(f"Expected tag file {tagfile} not found, skipping.")
                                continue
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
                    remote_source_tags = client.list_files(qlib, path="video_tags/source_tags/user", **content_args)
                except HTTPError:
                    logger.debug(f"No source tags found for {qwt}")
                    remote_source_tags = []
                
                for local_source in local_source_tags:
                    tagfile = os.path.join(config["storage"]["tags"], qhit, "external_tags", local_source)
                    file_jobs.append(ElvClient.FileJob(local_path=tagfile,
                                                        out_path=f"video_tags/source_tags/user/{local_source}",
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
            if os.path.exists(os.path.join(config["storage"]["tags"], qhit)):
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
            if "keeplasttemp" in os.environ.get("TAGGER_AGG", ""):
                ## for debugging, keep the last temp directory
                global last_tmpdir
                last_tmpdir = tmpdir
            else:
                tmpdir.cleanup()
        
        if not args.leave_open:
            client.finalize_files(qwt, qlib)

        client.set_commit_message(qwt, "uploaded/aggregated ML tags (taggerv2)", qlib)

        return Response(response=json.dumps({'message': 'Succesfully uploaded tag files. Please finalize the write token.', 'write token': qwt}), status=200, mimetype='application/json')
    
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

    def _startup():
        threading.Thread(target=_job_watcher, daemon=True).start()
        threading.Thread(target=_job_starter, args=("gpu", gpu_queue), daemon=True).start()
        threading.Thread(target=_job_starter, args=("cpu", cpu_queue), daemon=True).start()

    # handle shutdown signals
    signal.signal(signal.SIGINT, lambda sig, frame: _shutdown())
    signal.signal(signal.SIGTERM, lambda sig, frame: _shutdown())

    # in case of a different cause for shutdown other than a termination signal
    atexit.register(_shutdown)

    _startup()
    CORS(app)
    return app

LOCAL_CONFIG = "tagger-config.yml"
def main():
    if args.directory:
        os.chdir(args.directory)
        logger.info(f"changed directory to {args.directory}")
        
        if not os.path.exists(LOCAL_CONFIG):
            logger.error(f"You have specified directory {args.directory} but no {LOCAL_CONFIG} file was found there. This is probably an error.")
            sys.exit(1)
    if os.path.exists(LOCAL_CONFIG):
        reload_config(LOCAL_CONFIG)

    logger.info("Python interpreter version: " + sys.version)
    app = get_flask_app()

    serve(app, host=args.host, port=args.port)

if __name__ == '__main__':
    setproctitle.setproctitle("content-tagger")
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--directory', type=str)
    args = parser.parse_args()
    main()
