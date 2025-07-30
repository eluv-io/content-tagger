import argparse
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
from loguru import logger
import os
from collections import defaultdict
import threading
from requests.exceptions import HTTPError
import time
import signal
import atexit
import setproctitle
import sys
from waitress import serve
import traceback

from config import config, reload_config
from src.tagger.jobs import JobsStore
from src.api.tagging.handlers import handle_tag, handle_image_tag, handle_status, handle_stop
from src.api.tags.handlers import handle_finalize, handle_aggregate
from src.api.errors import BadRequestError, MissingResourceError

from src.tagger.tagger import Tagger
from src.tagger.resource_manager import ResourceManager

    
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
    def tag(qhit: str) -> Response:
        return handle_tag(qhit)
    
    @app.route('/<qhit>/image_tag', methods=['POST'])
    def image_tag(qhit: str) -> Response:
        return handle_image_tag(qhit)
    
    @app.route('/<qhit>/status', methods=['GET'])
    def status(qhit: str) -> Response:
        return handle_status(qhit)
    
    @app.route('/<qhit>/stop/<feature>', methods=['POST'])
    def stop(qhit: str, feature: str) -> Response:
        return handle_stop(qhit, feature)
    
    @app.route('/<qhit>/upload_tags', methods=['POST'])
    def upload(qhit: str) -> Response:
        return handle_upload(qhit)

def boot_state(app: Flask) -> None:
    app_state = {}

    app_state["filesystem_lock"] = threading.Lock()

    app_state["finalize_lock"] = defaultdict(threading.Lock)

    app_state["resource_manager"] = ResourceManager()

    app_state["jobs_store"] = JobsStore()

    app_state["tagger"] = Tagger(
        job_store=app_state["jobs_store"],
        manager=app_state["resource_manager"],
        filesystem_lock=app_state["filesystem_lock"],
    )

    app.config["state"] = app_state

def get_flask_app():
    app = Flask(__name__)
        
    @app.route('/<qhit>/upload_tags', methods=['POST'])
    def upload(qhit: str):
        handle_upload(qhit)

    @app.route('/<qhit>/write', methods=['POST'])
    @app.route('/<qhit>/finalize', methods=['POST'])
    def finalize(qhit: str) -> Response:
        handle_finalize(qhit)

    @app.route('/<qhit>/aggregate', methods=['POST'])
    def aggregate(qhit: str) -> Response:
        handle_aggregate(qhit)

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
