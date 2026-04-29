
import pytest
from requests import HTTPError

from src.api.auth import Authenticator
from src.common.content import Content


def test_auth(authenticator: Authenticator, q: Content):
    authenticator.authenticate(q)
    invalid = Content(qid=q.qid, token="invalid")
    with pytest.raises(HTTPError):
        try:
            authenticator.authenticate(invalid)
        except HTTPError as e:
            assert e.response.status_code < 500
            raise e