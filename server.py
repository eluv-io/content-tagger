import argparse
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS
import json
from requests.exceptions import HTTPError
import atexit
import signal
import setproctitle
import sys
from waitress import serve
import os

from src.service.impl.direct_api import DirectAPI
from src.service.impl.queue_based import QueueClient
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tags.tagstore.factory import create_tagstore
from src.fetch.factory import FetchFactory
from src.common.content import ContentFactory
from src.tag_containers.registry import ContainerRegistry
from src.tags.track_resolver import TrackResolver
from src.common.logging import logger

from src.api.tagging.handlers import handle_tag, handle_status, handle_status_content, handle_stop_model, handle_stop_content
from src.api.content_status.handlers import handle_content_status
from src.api.model_status.handlers import handle_model_status
from src.service.abstract import TagAPI
from src.tagging.fabric_tagging.queue.fs_jobstore import FsJobStore
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.tag_runner import TagRunner
from src.common.errors import *
from app_config import AppConfig

def configure_routes(app: Flask) -> None:
    # Configure the Flask app with the routes defined in this module.

    @app.errorhandler(BadRequestError)
    def handle_bad_request(e):
        logger.opt(exception=e).error("Got bad request error")
        return jsonify({'error': e.message}), 400

    @app.errorhandler(HTTPError)
    def handle_http_error(e):
        logger.opt(exception=e).error("Got HTTP error")
        status_code = e.response.status_code
        error_resp = json.loads(e.response.text)
        return jsonify({'code': status_code, 'error': error_resp}), status_code

    @app.errorhandler(MissingResourceError)
    def handle_missing_resource(e):
        logger.opt(exception=e).error("Missing resource error")
        return jsonify({'code': 404, 'message': e.message}), 404
    
    @app.errorhandler(ExternalServiceError)
    def handle_external_service_error(e):
        logger.opt(exception=e).error("External service error")
        return jsonify({'error': "An upstream service that tagging depends on is not available"}), 502

    @app.route('/<qhit>/tag', methods=['POST'])
    def tag(qhit: str) -> Response:
        return handle_tag(qhit)
    
    @app.route('/<qhit>/job-status', methods=['GET'])
    def status(qhit: str) -> Response:
        return handle_status_content(qhit)

    @app.route('/job-status', methods=['GET'])
    def status_all() -> Response:
        return handle_status()
    
    @app.route('/<qhit>/stop/<feature>', methods=['POST'])
    def stop_model(qhit: str, feature: str) -> Response:
        return handle_stop_model(qhit, feature)

    @app.route('/<qhit>/stop', methods=['POST'])
    def stop_content(qhit: str) -> Response:
        return handle_stop_content(qhit)

    @app.route('/<qhit>/tag-status', methods=['GET'])
    def content_status(qhit: str) -> Response:
        return handle_content_status(qhit)

    @app.route('/<qhit>/tag-status/<model>', methods=['GET'])
    def model_status(qhit: str, model: str) -> Response:
        return handle_model_status(qhit, model)

    @app.route('/docs', strict_slashes=False)
    def docs_route():
        return send_from_directory('docs/api', 'openapi.html')

def _build_fabric_tagger(cfg: AppConfig) -> FabricTagger:
    return FabricTagger(
        system_tagger=ContainerScheduler(cfg.system),
        fetcher=FetchFactory(cfg.fetcher, create_tagstore(cfg.tagstore)),
        cregistry=ContainerRegistry(cfg.container_registry),
        tagstore=create_tagstore(cfg.tagstore),
        cfg=cfg.tagger,
        track_resolver=TrackResolver(cfg.track_resolver),
    )


def create_app_direct(config: AppConfig) -> Flask:
    """Standalone mode: API handlers call FabricTagger directly."""
    app = Flask(__name__)

    fabric_tagger = _build_fabric_tagger(config)
    app.config["state"] = {
        "tagger": fabric_tagger,
        "service": DirectAPI(fabric_tagger),
        "content_factory": ContentFactory(config.content),
    }

    def shutdown():
        tagger: FabricTagger = app.config["state"]["tagger"]
        if not tagger.shutdown_requested:
            tagger.cleanup()

    atexit.register(shutdown)
    configure_routes(app)
    CORS(app)
    return app


def create_app_queue_based(config: AppConfig) -> Flask:
    """Queue-based mode: API handlers enqueue via QueueClient; TagRunner drives FabricTagger."""
    app = Flask(__name__)

    fabric_tagger = _build_fabric_tagger(config)
    content_factory = ContentFactory(config.content)
    job_store: JobStore = FsJobStore(config.jobstore.base_url)
    loop = TagRunner(fabric_tagger, job_store, content_factory, config.tag_runner)

    app.config["state"] = {
        "tagger": fabric_tagger,
        "service": QueueClient(job_store),
        "content_factory": content_factory,
        "job_store": job_store,
        "loop": loop,
    }

    loop.start()

    def shutdown():
        if not loop._shutdown.is_set():
            loop.stop()

    atexit.register(shutdown)
    configure_routes(app)
    CORS(app)
    return app

def main():
    logger.info("Python interpreter version: " + sys.version)

    if args.directory:
        os.chdir(args.directory)
        logger.info(f"changed directory to {args.directory}")

    cfg = AppConfig.from_yaml(args.config)

    if args.standalone:
        logger.info("starting in standalone mode")
        app = create_app_direct(cfg)
    else:
        logger.info("starting in queue-based mode")
        app = create_app_queue_based(cfg)

    def _handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down")
        sys.exit(0)  # raises SystemExit, which triggers atexit handlers

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    serve(app, host=args.host, port=args.port)

if __name__ == '__main__':
    setproctitle.setproctitle("content-tagger")
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--config', type=str, default="config.yml")
    parser.add_argument('--directory', type=str)
    parser.add_argument('--standalone', action='store_true', help='Run in standalone mode')
    args = parser.parse_args()
    main()
