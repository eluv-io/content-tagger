import argparse
from flask import Flask, jsonify, g, request
from flask_cors import CORS
from flask_smorest import Api
from apispec.ext.marshmallow import MarshmallowPlugin
from werkzeug.exceptions import HTTPException
import json
from requests.exceptions import HTTPError
import atexit
import signal
import setproctitle
import sys
from waitress.server import create_server
import os
import time
import uuid

from src.api.arg_resolver import ArgsResolver
from src.api.tenant_defaults import TenantDefaultsResolver
from src.api.auth import Authenticator
from src.service.impl.direct_api import DirectAPI
from src.service.impl.queue_based import QueueService
from src.service.job_poster import JobPoster
from src.status.get_info import UserInfoResolver
from src.status.service import TaggingStatusService
from src.tagging.scheduling.scheduler import ContainerScheduler
from src.tagging.fabric_tagging.tagger import TaggerWorker
from src.tags.tagstore.factory import create_tagstore
from src.tags.vectorstore.factory import VectorstoreFactory
from src.tagging.fabric_tagging.source_resolver import SourceResolver
from src.fetch.factory import FetchFactory
from src.common.content import QAPIFactory
from src.tag_containers.registry import ContainerRegistry
from src.tags.track_resolver import TrackResolver
from src.common.logging import logger, configure_logging

from src.api.tagging.handlers import tagging_blp
from src.api.content_status.handlers import content_status_blp
from src.api.tagging.model_param_schemas import MODEL_PARAM_SCHEMAS, model_params_component_name
from src.api.tagging.scope_schemas import SCOPE_SCHEMAS, ScopeSchema, scope_component_name
from src.tagging.fabric_tagging.queue.fs_jobstore import FsJobStore
from src.tagging.fabric_tagging.queue.abstract import JobStore
from src.tagging.tag_runner import TagRunner
from src.common.errors import *
from app_config import AppConfig

_SMOREST_DEFAULTS = {
    "API_TITLE": "Eluvio Tagger API",
    "API_VERSION": "1.0.0",
    "OPENAPI_VERSION": "3.2.0",
    # serve the spec + Swagger UI (replaces the old hand-maintained docs/api/openapi.html)
    "OPENAPI_URL_PREFIX": "/",
    "OPENAPI_JSON_PATH": "openapi.json",
    "OPENAPI_SWAGGER_UI_PATH": "/docs",
    "OPENAPI_SWAGGER_UI_URL": "https://cdn.jsdelivr.net/npm/swagger-ui-dist/",
    "API_SPEC_OPTIONS": {
        "servers": [{"url": "https://ai.contentfabric.io/tagging-live"}],
    },
}


def _register_error_handlers(app: Flask) -> None:
    """Map the domain exceptions to HTTP responses.

    flask-smorest handles request-validation errors (422) and other HTTPExceptions
    itself; these handlers cover the application's own exception types.
    """

    @app.errorhandler(BadRequestError)
    def handle_bad_request(e):
        logger.warning(f"Bad request: {e.message}")
        return jsonify({'error': e.message}), 400

    @app.errorhandler(HTTPError)
    def handle_http_error(e):
        status_code = e.response.status_code
        if status_code >= 500:
            logger.opt(exception=e).error("Upstream HTTP error")
        else:
            logger.warning(f"Upstream HTTP error: {e}")
        try:
            error_resp = json.loads(e.response.text)
        except ValueError:
            error_resp = e.response.text
        return jsonify({'code': status_code, 'error': error_resp}), status_code

    @app.errorhandler(MissingResourceError)
    def handle_missing_resource(e):
        logger.info(f"Missing resource: {e.message}")
        return jsonify({'code': 404, 'message': e.message}), 404

    @app.errorhandler(ForbiddenError)
    def handle_forbidden(e):
        logger.warning(f"Forbidden: {e.message}")
        return jsonify({'code': 403, 'message': e.message}), 403

    @app.errorhandler(ExternalServiceError)
    def handle_external_service_error(e):
        logger.opt(exception=e).error("External service error")
        return jsonify({'message': "An upstream service failed", 'error': str(e)}), 502

    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        # let flask-smorest / werkzeug handle HTTP errors (validation 422, 404, ...)
        if isinstance(e, HTTPException):
            return e.get_response()
        logger.opt(exception=e).error("Unhandled exception in API")
        return jsonify({'message': "An unexpected error occurred", 'error': str(e)}), 500


def _register_request_logging(app: Flask) -> None:
    """Log one line per request and tag every log line emitted while handling it with a request id."""

    @app.before_request
    def _start_request():
        g.request_id = (request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12])[:64]
        g.request_start = time.monotonic()
        g.log_ctx = logger.contextualize(request_id=g.request_id)
        g.log_ctx.__enter__()

    @app.after_request
    def _log_request(response):
        if "request_start" not in g:
            return response
        duration_ms = round((time.monotonic() - g.request_start) * 1000)

        quiet = request.method == "GET" and response.status_code < 400 and duration_ms < 1000

        logger.log(
            "DEBUG" if quiet else "INFO",
            "{} {} {} {}ms", request.method, request.path, response.status_code, duration_ms,
            remote_addr=request.headers.get("X-Forwarded-For", request.remote_addr),
        )
        response.headers["X-Request-ID"] = g.request_id
        return response

    @app.teardown_request
    def _end_request(_exc):
        ctx = g.pop("log_ctx", None)
        if ctx is not None:
            ctx.__exit__(None, None, None)


def configure_routes(app: Flask) -> None:
    # Configure the Flask app: error handlers, the flask-smorest Api, and the blueprints.
    for key, value in _SMOREST_DEFAULTS.items():
        app.config.setdefault(key, value)

    _register_error_handlers(app)
    _register_request_logging(app)

    api = Api(app)

    assert api.spec is not None

    # register the polymorphic scope variants as OpenAPI components so the `scope` field's
    # oneOf refs (see scope_oneof_metadata) resolve.
    api.spec.components.schema("Scope", schema=ScopeSchema)
    for scope_type, scope_schema in SCOPE_SCHEMAS.items():
        api.spec.components.schema(scope_component_name(scope_type), schema=scope_schema)

    # register the per-model params variants so the `model_params` oneOf refs
    # (see model_params_oneof_metadata) resolve.
    for model_name, params_schema in MODEL_PARAM_SCHEMAS.items():
        api.spec.components.schema(model_params_component_name(model_name), schema=params_schema)

    # same schemas, rendered inline for the /models listing
    converter = next(p for p in api.spec.plugins if isinstance(p, MarshmallowPlugin)).converter
    app.config["state"]["model_params_schemas"] = {
        model_name: converter.schema2jsonschema(params_schema)
        for model_name, params_schema in MODEL_PARAM_SCHEMAS.items()
    }

    api.register_blueprint(tagging_blp)
    api.register_blueprint(content_status_blp)

def _build_worker(cfg: AppConfig) -> TaggerWorker:
    qfactory = QAPIFactory(cfg.content)
    tagstore = create_tagstore(cfg.tagstore)
    vectorstores = VectorstoreFactory(cfg.vectorstore)
    track_resolver = TrackResolver(cfg.label_resolver, cfg.model_configs)
    model_configs = cfg.model_configs
    return TaggerWorker(
        system_tagger=ContainerScheduler(cfg.system),
        fetcher=FetchFactory(cfg.fetcher, create_tagstore(cfg.tagstore), qfactory),
        cregistry=ContainerRegistry(cfg.container_registry, model_configs),
        tagstore=tagstore,
        vectorstores=vectorstores,
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
    qfactory = QAPIFactory(config.content)
    tenant_defaults = TenantDefaultsResolver(UserInfoResolver(config.user_info_resolver), qfactory)
    arg_resolver = ArgsResolver(config.model_configs, qfactory, tenant_defaults)
    app.config["state"] = {
        "service": DirectAPI(worker),
        "status_service": TaggingStatusService(
            tagstore=worker.tagstore, 
            track_resolver=worker.track_resolver
        ),
        "authenticator": Authenticator(config.content.config_url),
        "arg_resolver": arg_resolver,
        # for listing API
        "model_configs": config.model_configs,
        "track_resolver": worker.track_resolver,
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
    tenant_defaults = TenantDefaultsResolver(user_info_resolver, qfactory)
    arg_resolver = ArgsResolver(config.model_configs, api_factory=qfactory, tenant_defaults=tenant_defaults)
    job_poster = JobPoster(job_store, worker.track_resolver, config.model_configs, qfactory)

    app.config["state"] = {
        "service": QueueService(job_poster=job_poster),
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
        "model_configs": config.model_configs,
        "track_resolver": worker.track_resolver,
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
    configure_logging(cfg.logging)

    if args.standalone:
        logger.info("starting in standalone mode")
        app = create_app_direct(cfg)
    else:
        logger.info("starting in queue-based mode")
        app = create_app_queue_based(cfg)

    server = create_server(app, host=args.host, port=args.port)

    def _handle_exit_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        server.close()  # finishes in-flight requests
        sys.exit(0) # raises SystemExit, which triggers atexit handlers

    signal.signal(signal.SIGTERM, _handle_exit_signal)
    signal.signal(signal.SIGINT, _handle_exit_signal)

    if not args.standalone:
        loop = app.config["state"]["loop"]
        def _handle_sighup(signum, frame):
            logger.info("Received SIGHUP, entering quiesce mode")
            loop.quiesce()
        signal.signal(signal.SIGHUP, _handle_sighup)
    else:
        # In standalone mode, SIGHUP behaves like SIGTERM
        signal.signal(signal.SIGHUP, _handle_exit_signal)

    server.run()

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
