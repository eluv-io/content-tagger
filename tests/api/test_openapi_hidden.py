from flask import Flask

from server import configure_routes
from src.api.tagging.request_format import StartJobsRequestSchema


def _spec() -> dict:
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["state"] = {}
    configure_routes(app)
    response = app.test_client().get("/openapi.json")
    assert response.status_code == 200
    return response.get_json()


def test_priority_not_in_openapi_docs():
    spec = _spec()
    options = spec["components"]["schemas"]["TaggerOptions"]["properties"]
    assert "replace" in options  # sanity: the schema is rendered, just without priority
    assert "priority" not in options
    assert "priority" not in str(spec).lower()


def test_priority_still_accepted():
    req = StartJobsRequestSchema().load({
        "options": {"priority": 5},
        "jobs": [{"model": "asr", "overrides": {"priority": 50}}],
    })
    assert req.options.priority == 5
    assert req.jobs[0].overrides.priority == 50
