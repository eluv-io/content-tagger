
from typing import Dict

import requests
from elv_client_py import ElvClient
from flask import Request, Response, make_response
from requests.exceptions import HTTPError

from src.common.errors import BadRequestError
from src.common.content import Content, parse_qhit

class Authenticator:
    def __init__(self, config_url: str):
        self.config_url = config_url

    def authenticate(self, q: Content) -> None:
        """Basic authentication against the content. Returns None if successful, raises HTTPError if authentication fails."""
        try:
            client = ElvClient.from_configuration_url(config_url=self.config_url, static_token=q.token)
            client.content_object(**parse_qhit(q.qid))
        except HTTPError as e:
            raise HTTPError(
                f"Failed to access the requested content with the provided authorization: {q.qid}") from e

def get_authorization(req: Request) -> str:
    """Get the authorization token from the request headers or query parameters."""
    auth = req.headers.get('Authorization', None) or req.args.get(
        'authorization', None)
    if not auth:
        raise BadRequestError("Authorization token is required")
    return auth

def convert_response(resp: requests.Response) -> Response:
    flask_response = make_response(resp.content)
    flask_response.status_code = resp.status_code
    return flask_response