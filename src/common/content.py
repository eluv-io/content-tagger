from dataclasses import dataclass
from typing import Any
import base64
import dacite
import json
import requests

from elv_client_py import ElvClient

from src.common.errors import BadRequestError

@dataclass
class ContentConfig:
    config_url: str
    parts_url: str
    live_media_url: str

class Content:
    """Content object representation and API wrapper.

    Serves the following uses:
        - Caches qinfo call (qid, qhash, qlib) so that they don't need to be requeried
        - Bridges the the parts and regular fabric APIs into one interface
        - Helps reduce parameter counts in functions by not requiring the qhit & the ElvClient to
            be passed separately
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
        
        live_client = ElvClient.from_configuration_url(
            cfg.live_media_url, static_token=auth)

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
        self._live_client = live_client

        self.cfg = cfg

    def get_child(self, qhit: str) -> "Content":
        """Get a child content object as a Content instance."""
        return Content(qhit, self._client.token, self.cfg)

    def token(self) -> str:
        return self._client.token

    def content_object_versions(self) -> dict[str, Any]:
        """Get all versions of the content object."""
        return self._client.content_object_versions(object_id=self.qid, library_id=self.qlib)
    
    def download_part(self, **kwargs) -> None:
        """Download a part of the content object."""
        if self.qwt:
            kwargs["write_token"] = self.qwt
        else:
            kwargs["version_hash"] = self.qhash
        return self._parts_client.download_part(library_id=self.qlib, **kwargs)
    
    def live_media_segment(
        self,
        object_id: str,
        dest_path: str,
        segment_idx: int | None = None, 
        segment_length: int = 4,
        stream: str = ""
    ) -> ElvClient.LiveMediaSegment:
        url = self._live_client.fabric_uris[0]
        url = '/'.join([url, 'q', object_id, 'rep', 'media', 'segment'])
        resp = requests.get(
            url, 
            params={
                "authorization": self._client.token,
                "num": segment_idx,
                "duration": segment_length,
                "stream": stream
            }, 
            stream=True
        )
        if resp.status_code == 200:
            with open(dest_path, "wb") as file:
                for chunk in resp.iter_content(chunk_size=8192):
                    file.write(chunk)
        else:
            resp.raise_for_status()
        info = resp.headers.get('X-Content-Fabric-Segment-Info')
        # base64 decode
        if info is None:
            raise ValueError("No segment info in response")
        info = base64.b64decode(info).decode('utf-8')
        segment_info = dacite.from_dict(
            data_class=ElvClient.LiveMediaSegment,
            data=json.loads(info)
        )
        return segment_info

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
    # Useful for dependency injection to avoid having to pass cfg around everywhere
    def __init__(self, cfg: ContentConfig):
        self.cfg = cfg

    def create_content(self, qhit: str, auth: str) -> Content:
        return Content(qhit, auth, self.cfg)

def parse_qhit(qhit: str) -> dict[str, str]:
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