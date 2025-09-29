import argparse
from flask import Flask, Response, jsonify
from flask_cors import CORS
import json
from loguru import logger
from requests.exceptions import HTTPError
import atexit
import setproctitle
import sys
from waitress import serve

from src.tagger.system_tagging.resource_manager import SystemTagger
from src.tagger.fabric_tagging.tagger import FabricTagger
from src.tags.tagstore.factory import create_tagstore
from src.fetch.fetch_video import Fetcher
from src.common.content import ContentFactory
from src.tag_containers.containers import ContainerRegistry
from src.tags.conversion import TagConverter

from src.api.tagging.handlers import handle_tag, handle_image_tag, handle_status, handle_stop
from src.api.upload.handlers import handle_commit
from src.common.errors import BadRequestError, MissingResourceError
from app_config import AppConfig

def configure_routes(app: Flask) -> None:
    # Configure the Flask app with the routes defined in this module.

    @app.errorhandler(BadRequestError)
    def handle_bad_request(e):
        logger.exception(f"Bad request: {e}")
        return jsonify({'error': e.message}), 400

    @app.errorhandler(HTTPError)
    def handle_http_error(e):
        logger.exception(f"HTTP error: {e}")
        status_code = e.response.status_code
        error_resp = json.loads(e.response.text)
        return jsonify({'message': 'Fabric API error', 'error': error_resp}), status_code

    @app.errorhandler(MissingResourceError)
    def handle_missing_resource(e):
        logger.exception(f"Missing resource: {e}")
        return jsonify({'message': e.message}), 404

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

    @app.route('/<qhit>/commit', methods=['POST'])
    def commit(qhit: str) -> Response:
        return handle_commit(qhit)

    #@app.route('/<qhit>/write', methods=['POST'])
    #@app.route('/<qhit>/finalize', methods=['POST'])
    #def finalize(qhit: str) -> Response:
    #    return handle_finalize(qhit)
#
    #@app.route('/<qhit>/aggregate', methods=['POST'])
    #def aggregate(qhit: str) -> Response:
    #    return handle_aggregate(qhit)

def boot_state(app: Flask, cfg: AppConfig) -> None:
    app_state = {}

    system_tagger = SystemTagger(cfg.system)
    tagstore = create_tagstore(cfg.tagstore)
    fetcher = Fetcher(cfg.fetcher, tagstore)
    container_registry = ContainerRegistry(cfg.container_registry)

    app_state["tagger"] = FabricTagger(
        system_tagger=system_tagger,
        fetcher=fetcher,
        cregistry=container_registry,
        tagstore=tagstore
    )

    app_state["content_factory"] = ContentFactory(cfg.content)

    app_state["tag_converter"] = TagConverter(cfg.tag_converter)

    app.config["state"] = app_state

def configure_lifecycle(app: Flask) -> None:

    def shutdown():
        app_state = app.config["state"]
        tagger: FabricTagger = app_state["tagger"]
        if tagger.shutdown_requested is False:
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
    cfg = AppConfig.from_yaml(args.config)
    app = create_app(cfg)

    serve(app, host=args.host, port=args.port)

if __name__ == '__main__':
    setproctitle.setproctitle("content-tagger")
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--config', type=str, default="config.yml")
    args = parser.parse_args()
    main()
