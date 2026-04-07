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

from src.api.arg_resolver import ArgsResolver
from src.api.auth import Authenticator
from src.service.impl.direct_api import DirectAPI
from src.service.impl.queue_based import QueueService
from src.status.get_info import UserInfoResolver
from src.status.service import TaggingStatusService
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.tagging.fabric_tagging.tagger import TaggerWorker
from src.tags.tagstore.factory import create_tagstore
from src.tagging.fabric_tagging.source_resolver import SourceResolver
from src.fetch.factory import FetchFactory
from src.common.content import QAPIFactory
from src.tag_containers.registry import ContainerRegistry
from src.tags.track_resolver import TrackResolver
from src.common.logging import logger

from src.api.tagging.handlers import handle_tag, handle_status, handle_status_content, handle_stop_model, handle_stop_content
from src.api.content_status.handlers import handle_content_status, handle_model_status
from src.api.extension.handlers import handle_list_models, handle_delete_job
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

    @app.errorhandler(ForbiddenError)
    def handle_forbidden(e):
        logger.opt(exception=e).error("Forbidden error")
        return jsonify({'code': 403, 'message': e.message}), 403
    
    @app.errorhandler(ExternalServiceError)
    def handle_external_service_error(e):
        logger.opt(exception=e).error("External service error")
        return jsonify({'message': "An upstream service failed", 'error': str(e)}), 502
    
    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        logger.opt(exception=e).error("Unhandled exception in API")
        return jsonify({'message': "An unexpected error occurred", 'error': str(e)}), 500

    @app.route('/<qid>/tag', methods=['POST'])
    def tag(qid: str) -> Response:
        return handle_tag(qid)
    
    @app.route('/<qid>/job-status', methods=['GET'])
    def status(qid: str) -> Response:
        return handle_status_content(qid)

    @app.route('/job-status', methods=['GET'])
    def status_all() -> Response:
        return handle_status()
    
    @app.route('/<qid>/stop/<feature>', methods=['POST'])
    def stop_model(qid: str, feature: str) -> Response:
        return handle_stop_model(qid, feature)

    @app.route('/<qid>/stop', methods=['POST'])
    def stop_content(qid: str) -> Response:
        return handle_stop_content(qid)

    @app.route('/<qid>/tag-status', methods=['GET'])
    def content_status(qid: str) -> Response:
        return handle_content_status(qid)

    @app.route('/<qid>/tag-status/<model>', methods=['GET'])
    def model_status(qid: str, model: str) -> Response:
        return handle_model_status(qid, model)

    @app.route('/models', methods=['GET'])
    def list_models_route() -> Response:
        return handle_list_models()

    @app.route('/jobs/<job_id>', methods=['DELETE'])
    def delete_job_route(job_id: str) -> Response:
        return handle_delete_job(job_id)

    @app.route('/docs', strict_slashes=False)
    def docs_route():
        return send_from_directory('docs/api', 'openapi.html')

def _build_worker(cfg: AppConfig) -> TaggerWorker:
    qfactory = QAPIFactory(cfg.content)
    tagstore = create_tagstore(cfg.tagstore)
    track_resolver = TrackResolver(cfg.track_resolver)
    return TaggerWorker(
        system_tagger=ContainerScheduler(cfg.system),
        fetcher=FetchFactory(cfg.fetcher, create_tagstore(cfg.tagstore), qfactory),
        cregistry=ContainerRegistry(cfg.container_registry),
        tagstore=tagstore,
        cfg=cfg.tagger,
        track_resolver=track_resolver,
        source_resolver=SourceResolver(create_tagstore(cfg.tagstore), track_resolver=track_resolver)
    )


def create_app_direct(config: AppConfig) -> Flask:
    """
    Development tagger API - this does not support all handlers.

    Standalone mode: API handlers call TaggerWorker directly."""
    app = Flask(__name__)

    worker = _build_worker(config)
    arg_resolver = ArgsResolver(worker.cregistry, QAPIFactory(config.content))
    app.config["state"] = {
        "service": DirectAPI(worker),
        "status_service": TaggingStatusService(
            tagstore=worker.tagstore, 
            track_resolver=worker.track_resolver
        ),
        "authenticator": Authenticator(config.content.config_url),
        "arg_resolver": arg_resolver,
        "worker": worker,  # Expose worker for testing purposes
    }

    def shutdown():
        if not worker.shutdown_requested:
            worker.cleanup()

    atexit.register(shutdown)
    configure_routes(app)
    CORS(app)
    return app


def create_app_queue_based(config: AppConfig) -> Flask:
    """
    Production tagger API
    
    Queue-based mode: API handlers enqueue via QueueService; TagRunner drives TaggerWorker."""
    app = Flask(__name__)

    worker = _build_worker(config)
    user_info_resolver = UserInfoResolver(config.user_info_resolver)
    job_store: JobStore = FsJobStore(config.jobstore.base_url, user_info_resolver=user_info_resolver)
    qfactory = QAPIFactory(config.content)
    arg_resolver = ArgsResolver(worker.cregistry, api_factory=qfactory)

    app.config["state"] = {
        "service": QueueService(job_store, qfactory),
        "status_service": TaggingStatusService(
            tagstore=worker.tagstore, 
            track_resolver=worker.track_resolver
        ),
        "arg_resolver": arg_resolver,
        "user_info_resolver": user_info_resolver,
        "authenticator": Authenticator(config.content.config_url),
        # for delete jobs endpoint
        "jobstore": job_store,
        # for listing API
        "container_registry": worker.cregistry,
        "worker": worker,  # Expose worker for testing purposes
    }

    loop = TagRunner(worker, job_store, config.tag_runner)
    app.config["state"]["loop"] = loop

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
