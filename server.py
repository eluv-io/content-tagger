import argparse
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS
import json
from requests.exceptions import HTTPError
import atexit
import setproctitle
import sys
from waitress import serve
import os

from src.api.tagging.impl.direct_api import DirectAPI
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tags.tagstore.factory import create_tagstore
from src.fetch.factory import FetchFactory
from src.common.content import ContentFactory
from src.tag_containers.registry import ContainerRegistry
from src.tags.track_resolver import TrackResolver
from src.common.logging import logger

from src.api.tagging.handlers import handle_tag, handle_status, handle_stop_model, handle_stop_content
from src.api.content_status.handlers import handle_content_status
from src.api.model_status.handlers import handle_model_status
from src.api.tagging.abstract import TagAPI
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
        return handle_status(qhit)
    
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

def boot_state(app: Flask, cfg: AppConfig) -> None:
    app_state = {}

    system_tagger = ContainerScheduler(cfg.system)
    tagstore = create_tagstore(cfg.tagstore)
    fetcher = FetchFactory(cfg.fetcher, tagstore)
    container_registry = ContainerRegistry(cfg.container_registry)
    track_resolver = TrackResolver(cfg.track_resolver)

    fabric_tagger = FabricTagger(
        system_tagger=system_tagger,
        fetcher=fetcher,
        cregistry=container_registry,
        tagstore=tagstore,
        cfg=cfg.tagger,
        track_resolver=track_resolver,
    )

    app_state["tagger"] = fabric_tagger

    app_state["service"] = DirectAPI(fabric_tagger)

    app_state["content_factory"] = ContentFactory(cfg.content)

    app.config["state"] = app_state

def configure_lifecycle(app: Flask) -> None:

    def shutdown():
        app_state = app.config["state"]
        tagger: TagAPI = app_state["service"]
        if tagger.shutdown_requested() is False:
            tagger.cleanup()

    atexit.register(shutdown)

def create_app(config: AppConfig) -> Flask:
    """Main entry point for the server."""
    app = Flask(__name__)
    boot_state(app, config)
    configure_routes(app)
    configure_lifecycle(app)
    CORS(app)
    return app

def main():
    logger.info("Python interpreter version: " + sys.version)

    if args.directory:
        os.chdir(args.directory)
        logger.info(f"changed directory to {args.directory}")

    cfg = AppConfig.from_yaml(args.config)
    app = create_app(cfg)

    serve(app, host=args.host, port=args.port)

if __name__ == '__main__':
    setproctitle.setproctitle("content-tagger")
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--config', type=str, default="config.yml")
    parser.add_argument('--directory', type=str)
    args = parser.parse_args()
    main()
