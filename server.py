import argparse
from flask import Flask, Response, jsonify
from flask_cors import CORS
import json
from loguru import logger
import os
from collections import defaultdict
import threading
from requests.exceptions import HTTPError
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
    

    @app.route('/<qhit>/write', methods=['POST'])
    @app.route('/<qhit>/finalize', methods=['POST'])
    def finalize(qhit: str) -> Response:
        handle_finalize(qhit)

    @app.route('/<qhit>/aggregate', methods=['POST'])
    def aggregate(qhit: str) -> Response:
        handle_aggregate(qhit)

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

def configure_lifecycle(app: Flask) -> None:

    def _cleanup():
        """Cleanup resources before shutdown."""
        logger.info("Cleaning up resources...")
        app_state = app.config["state"]
        app_state["resource_manager"].cleanup()
        app_state["jobs_store"].cleanup()
        logger.info("Cleanup completed.")
        os._exit(0)

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda signum, frame: _cleanup())
    signal.signal(signal.SIGTERM, lambda signum, frame: _cleanup())

def create_app() -> Flask:
    """Main entry point for the server."""
    app = Flask(__name__)
    boot_state(app)
    configure_routes(app)
    configure_lifecycle(app)
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
    app = create_app()

    serve(app, host=args.host, port=args.port)

if __name__ == '__main__':
    setproctitle.setproctitle("content-tagger")
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--directory', type=str)
    args = parser.parse_args()
    main()
