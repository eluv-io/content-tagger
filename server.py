import argparse
from typing import List, Optional
from flask import Flask, request, Response, Request
from flask_cors import CORS
from podman import PodmanClient
import json
from loguru import logger
from dataclasses import dataclass
import os
from elv_client_py import ElvClient
import shutil
from requests.exceptions import HTTPError

from config import config
from src.containers import run_container
from src.fetch import fetch_stream, StreamNotFoundError

@dataclass
class TagArgs:
    features: List[str]
    start_time: Optional[int]=None
    end_time: Optional[int]=None
    stream: Optional[str]=None
    authorization: Optional[str]=None

def get_flask_app():
    app = Flask(__name__)

    @app.route('/list', methods=['GET'])
    def list_services() -> Response:
        res = _list_services()    
        return Response(response=json.dumps(res), status=200, mimetype='application/json')
    
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
        for feature in args.features:
            if not args.stream:
                stream_name = config["services"][feature]["type"]
            else:
                stream_name = args.stream
            save_path = os.path.join(config["storage"]["parts"], qid)
            logger.info(f"Fetching parts to run {feature} on {qid}")
            try:
                part_paths = fetch_stream(qid, stream_name, os.path.join(save_path, stream_name), elv_client, args.start_time, args.end_time)
            except StreamNotFoundError:
                if not args.stream:
                    return Response(response=json.dumps({'error': f"Default stream {stream_name} not found for {qid}. Please use the `stream` argument to specify a valid stream."}), status=404, mimetype='application/json')
                else:
                    return Response(response=json.dumps({'error': f"Stream {args.stream} not found for {qid}"}), status=404, mimetype='application/json')
            except HTTPError as e:
                return Response(response=json.dumps({'error': str(e)}), status=403, mimetype='application/json')
            logger.info(f"Tagging {qid} with {feature}")
            with PodmanClient() as podman_client:
                try:
                    tag_files = run_container(podman_client, feature, part_paths)
                except Exception as e:
                    return Response(response=json.dumps({'error': str(e)}), status=500, mimetype='application/json')
            out_path = os.path.join(config["storage"]["tags"], qid, feature)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for f in tag_files:
                shutil.move(f, os.path.join(config["storage"]["tags"], qid, feature, os.path.basename(f)))
            
        return Response(response=json.dumps({'success': True}), status=200, mimetype='application/json')
    
    @app.route('/<qid>/finalize', methods=['POST'])
    def finalize(qid: str) -> Response:
        auth = _get_authorization(request)
        elv_client = ElvClient.from_configuration_url(config_url=config["fabric"]["config_url"], static_token=auth)
        qwt = request.args.get('write_token')
        qlib = elv_client.content_object_library_id(qid)
        jobs = []
        for feature in os.listdir(os.path.join(config["storage"]["tags"], qid)):
            for tag in os.listdir(os.path.join(config["storage"]["tags"], qid, feature)):
                jobs.append(ElvClient.FileJob(local_path=os.path.join(config["storage"]["tags"], qid, feature, tag), 
                                              out_path=f"video_tags/{feature}/{tag}",
                                              mime_type="application/json"))
        try:
            elv_client.upload_files(qwt, qlib, jobs)
        except HTTPError as e:
            return Response(response=json.dumps({'error': str(e)}), status=403, mimetype='application/json')
        
        return Response(response=json.dumps({'message': 'Succesfully uploaded tag files. Please finalize the write token.', 'write token': qwt}), status=200, mimetype='application/json')
    
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
    def _get_authorization(req: Request) -> str:
        auth = req.headers.get('Authorization')
        if auth:
            return auth
        return req.args.get('authorization')

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