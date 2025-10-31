
from typing import Dict

import requests
from elv_client_py import ElvClient
from flask import Request, Response, make_response
from requests.exceptions import HTTPError

from src.common.errors import BadRequestError
from src.common.content import Content, parse_qhit


def authenticate(client: ElvClient, qhit: str) -> None:
    """Basic authentication against the object. Returns None if successful, raises HTTPError if authentication fails."""
    try:
        client.content_object(**parse_qhit(qhit))
    except HTTPError as e:
        raise HTTPError(
            f"Failed to access the requested content with the provided authorization: {qhit}") from e


def get_authorization(req: Request) -> str:
    """Get the authorization token from the request headers or query parameters. Throws BadRequestError if not found."""
    auth = req.headers.get('Authorization', None) or req.args.get(
        'authorization', None)
    if not auth:
        raise BadRequestError("Authorization token is required")
    return auth


def get_client(req: Request, qhi: str, config_url: str) -> ElvClient:
    """Get an ElvClient instance from the request and qhit.
    This will raise a BadRequestError if the authorization token is not provided or HTTPError if the authentication fails.
    """
    auth = get_authorization(req)
    client = ElvClient.from_configuration_url(
        config_url=config_url, static_token=auth)
    # will throw http error if bad authentication, server will handle this and return the appropriate response
    authenticate(client, qhi)
    return client

def convert_response(resp: requests.Response) -> Response:
    flask_response = make_response(resp.content)
    flask_response.status_code = resp.status_code
    return flask_response

def is_same_auth_ctx(q: Content, other_qhit: str) -> bool:
    """Check if the other_qhit belongs to the same authorization context as q."""
    try:
        q.content_object(**parse_qhit(other_qhit))
    except HTTPError:
        return False

    return True