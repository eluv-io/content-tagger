import argparse
from typing import List, Optional, Literal
from flask import Flask, request, Response, Request
from flask_cors import CORS
from podman import PodmanClient
import json
from loguru import logger
from dataclasses import dataclass, asdict
import os
from elv_client_py import ElvClient
from collections import defaultdict
import threading
from requests.exceptions import HTTPError
import traceback
import time

from config import config
from src.fetch import fetch_stream, StreamNotFoundError
from src.manager import ResourceManager

@dataclass
class JobStatus:
    status: Literal["Starting", "Running", "Completed", "Failed", "Stopped by user"]
    tag_job_id: Optional[str]=None
    error: Optional[str]=None
    message: Optional[str]=None

def get_flask_app():
    app = Flask(__name__)
    manager = ResourceManager()

    lock = threading.Lock()
    active_jobs = defaultdict(dict)
    inactive_jobs = defaultdict(dict)

    @app.route('/list', methods=['GET'])
    def list_services() -> Response:
        res = _list_services()    
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
    
    @dataclass
    class TagArgs:
        features: List[str]
        start_time: Optional[int]=None
        end_time: Optional[int]=None
        stream: Optional[str]=None
        authorization: Optional[str]=None

    @app.route('/<qid>/tag', methods=['POST'])
    def tag(qid: str) -> Response:
        logger.debug(f"Request: {request.json}")
        data = request.json
        try:
            args = TagArgs(**data)
        except TypeError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        services = _list_services()
        for feature in args.features:
            if feature not in services:
                return Response(response=json.dumps({'error': f"Service {feature} not found"}), status=404, mimetype='application/json')
            
        auth = _get_authorization(request)
        elv_client = ElvClient.from_configuration_url(config_url=config["fabric"]["parts_url"], static_token=auth)
        with lock:
            for feature in args.features:
                if active_jobs[qid].get(feature, None):
                    return Response(response=json.dumps({'error': f"Tagging {feature} already in progress for {qid}"}), status=400, mimetype='application/json')
            for feature in args.features:
                active_jobs[qid][feature] = JobStatus(status="Starting")
            threading.Thread(target=_tag, args=(args.features, qid, elv_client, args.stream, args.start_time, args.end_time)).start()
        return Response(response=json.dumps({'message': f'Tagging started on {qid}'}), status=200, mimetype='application/json')
    
    # Download parts and then tag them
    def _tag(features: List[str], qid: str, elv_client: ElvClient, stream: Optional[str]=None, start_time: Optional[int]=None, end_time: Optional[int]=None) -> None:
        feature_to_parts = {}
        for feature in features:
            job = active_jobs[qid][feature]
            job.status = "Fetching parts"
            if not stream:
                stream_name = config["services"][feature]["type"]
            else:
                stream_name = stream
            save_path = os.path.join(config["storage"]["parts"], qid)
            logger.info(f"Fetching parts to run {feature} on {qid}")
            try:
                part_paths = fetch_stream(qid, stream_name, os.path.join(save_path, stream_name), elv_client, start_time, end_time)
            except StreamNotFoundError:
                with lock:
                    job = active_jobs[qid][feature]
                    job.status = "failed"
                    if not args.stream:
                        job.error = f"Stream {stream_name} not found for {qid}. Please specify a stream."
                    else:
                        job.error = f"Stream {stream_name} not found for {qid}"
                    inactive_jobs[qid][feature] = job
                    del active_jobs[qid][feature]
            except HTTPError as e:
                with lock:
                    job = active_jobs[qid][feature]
                    job.status = "failed"
                    job.error = f"Failed to fetch stream {stream_name} for {qid}: {str(e)}"
                    inactive_jobs[qid][feature] = job
                    del active_jobs[qid][feature]
            feature_to_parts[feature] = part_paths
        
        for feature, part_paths in feature_to_parts.items():
            logger.info(f"Tagging {qid} with {feature}")
            with lock:
                job = active_jobs[qid][feature]
                try:
                    job_id = manager.run(feature, part_paths)
                    job.tag_job_id = job_id
                    job.status = "Tagging parts"
                except Exception as e:
                    logger.error(f"Failed to run {feature} on {qid}: {traceback.format_exc()})")
                    job.status = "failed"
                    job.error = str(e)
                    inactive_jobs[qid][feature] = job
                    del active_jobs[qid][feature]

        running_features = list(active_jobs[qid].keys())
        # watch jobs
        while len(running_features) > 0:
            for feature in running_features:
                job = active_jobs[qid][feature]
                status = manager.status(job.tag_job_id)
                if status.status == "running":
                    continue
                job.status = status.status # "completed" or "failed"
                inactive_jobs[qid][feature] = job
                del active_jobs[qid][feature]
                running_features.remove(feature)
            time.sleep(5)
        # done tagging

    @dataclass
    class FinalizeArgs:
        write_token: str
        force: bool=False
    
    @app.route('/<qid>/finalize', methods=['POST'])
    def finalize(qid: str) -> Response:
        try:
            args = FinalizeArgs(**request.json)
        except TypeError as e:
            return Response(response=json.dumps({'message': 'invalid request', 'error': str(e)}), status=400, mimetype='application/json')
        auth = _get_authorization(request)
        if not auth:
            return Response(response=json.dumps({'error': 'No authorization provided'}), status=400, mimetype='application/json')
        elv_client = ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=auth)
        qwt = args.write_token
        try:
            qlib = elv_client.content_object_library_id(qid)
        except HTTPError as e:
            return Response(response=json.dumps({'error': str(e)}), status=403, mimetype='application/json')
        jobs_running = sum(len(active_jobs[qid][feature]) for feature in active_jobs[qid])
        if jobs_running > 0 and not args.force:
            return Response(response=json.dumps({'error': 'Some jobs are still running. Use the `force` parameter to finalize anyway.'}), status=400, mimetype='application/json')

        with lock():
            jobs = []
            for stream in os.listdir(os.path.join(config["storage"]["tags"], qid)):
                for feature in os.listdir(os.path.join(config["storage"]["tags"], qid, stream)):
                    for tag in os.listdir(os.path.join(config["storage"]["tags"], qid, stream, feature)):
                        jobs.append(ElvClient.FileJob(local_path=os.path.join(config["storage"]["tags"], qid, stream, feature, tag), 
                                                    out_path=f"video_tags/{stream}/{feature}/{tag}",
                                                    mime_type="application/json"))
        try:
            elv_client.upload_files(qwt, qlib, jobs)
        except HTTPError as e:
            return Response(response=json.dumps({'error': str(e)}), status=403, mimetype='application/json')
        
        return Response(response=json.dumps({'message': 'Succesfully uploaded tag files. Please finalize the write token.', 'write token': qwt}), status=200, mimetype='application/json')
    
    # get status of all jobs for a qid
    @app.route('/<qid>/status', methods=['GET'])
    def status(qid: str) -> Response:
        client = ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=_get_authorization(request))
        if not _authenticate(client, qid):
            return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')

        features = set(active_jobs[qid].keys()) | set(inactive_jobs[qid].keys())
        if len(features) == 0:
            return Response(response=json.dumps({'error': f"No jobs started for {qid}"}), status=404, mimetype='application/json')
        res = {feature: asdict(active_jobs[qid][feature]) if feature in active_jobs[qid] else asdict(inactive_jobs[qid][feature]) for feature in features}
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
    
    @app.route('/<qid>/stop/<feature>', methods=['DELETE'])
    def stop(qid: str, feature: str) -> Response:
        client = ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=_get_authorization(request))
        if not _authenticate(client, qid):
            return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
        with lock:
            if feature not in active_jobs[qid]:
                return Response(response=json.dumps({'error': f"No job running for {feature} on {qid}"}), status=404, mimetype='application/json')
            job = active_jobs[qid][feature]
            if job.tag_job_id:
                # TODO: very unlikely race condition if job starts in between these two lines
                manager.stop(job.tag_job_id)
            job.status = "Stopped by user"
            job.tag_job_id = None
            inactive_jobs[qid][feature] = job
            del active_jobs[qid][feature]
        return Response(response=json.dumps({'message': f"Stopped {feature} on {qid}"}), status=200, mimetype='application/json')
    
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
    
    # authentication can be in header or query string
    # returns None if no authorization is found
    def _get_authorization(req: Request) -> Optional[str]:
        auth = req.headers.get('Authorization', None)
        if auth:
            return auth
        return req.args.get('authorization', None)
    
    def _authenticate(client: ElvClient, qid: str) -> bool:
        try:
            client.content_object(qid)
        except HTTPError as e:
            return False
        return True
            
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