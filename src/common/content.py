from dataclasses import dataclass
from typing import Any, Dict

from elv_client_py import ElvClient

from src.api.auth import parse_qhit

@dataclass
class ContentConfig:
    config_url: str
    parts_url: str

class Content():
    """Content object representation and API wrapper.
    """

    def __init__(
            self, 
            qhit: str,
            auth: str,
            cfg: ContentConfig
    ):
        client = ElvClient.from_configuration_url(
            cfg.config_url, static_token=auth)

        parts_client = ElvClient.from_configuration_url(
            cfg.parts_url, static_token=auth)

        # will raise HTTPError if auth is invalid or qhit is not found
        qinfo = client.content_object(**parse_qhit(qhit))

        self.qid = qinfo["id"]
        self.qhash = qinfo.get("hash", None)
        self.qwt = qinfo.get("write_token", None)

        assert self.qhash or self.qwt, f"Content object must have either a hash or a write token. {qinfo}"

        self.qlib = qinfo["qlib_id"]
        self.qhit = qhit
        self._client = client
        self._parts_client = parts_client

    def token(self) -> str:
        return self._client.token

    def content_object_versions(self) -> Dict[str, Any]:
        """Get all versions of the content object."""
        return self._client.content_object_versions(object_id=self.qid, library_id=self.qlib)
    
    def download_part(self, **kwargs) -> None:
        """Download a part of the content object."""
        if self.qwt:
            kwargs["write_token"] = self.qwt
        else:
            kwargs["version_hash"] = self.qhash
        return self._parts_client.download_part(library_id=self.qlib, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            raise AttributeError(
                f"'{name}' Content type does not have this attribute.")
        if callable(attr):
            def wrapper(*args, **kwargs):
                if self.qwt:
                    kwargs["write_token"] = self.qwt
                else:
                    kwargs["version_hash"] = self.qhash
                return attr(*args, library_id=self.qlib, **kwargs)
            return wrapper
        return attr

    def __str__(self):
        return f"Content(qhit={self.qhit}, qid={self.qid}, qhash={self.qhash}, qlib={self.qlib})"

class ContentFactory:
    def __init__(self, cfg: ContentConfig):
        self.cfg = cfg

    def create_content(self, qhit: str, auth: str) -> Content:
        return Content(qhit, auth, self.cfg)
