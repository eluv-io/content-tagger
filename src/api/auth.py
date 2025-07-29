
from typing import Dict

import requests
from elv_client_py import ElvClient
from flask import Request, Response, make_response
from requests.exceptions import HTTPError

from src.api.errors import BadRequestError


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


def parse_qhit(qhit: str) -> Dict[str, str]:
    """Parse a qhit into a dictionary so it can be passed to elv_client_py functions 
    and use the correct argument."""
    if not isinstance(qhit, str):
        raise BadRequestError(f"qhit must be a string, got {type(qhit)}")

    if qhit.startswith("hq__"):
        return {"version_hash": qhit}
    elif qhit.startswith("tqw__"):
        return {"write_token": qhit}
    elif qhit.startswith("iq__"):
        return {"object_id": qhit}

    raise BadRequestError(f"Invalid qhit: {qhit}")


def convert_response(resp: requests.Response) -> Response:
    flask_response = make_response(resp.content)
    flask_response.status_code = resp.status_code
    return flask_response
