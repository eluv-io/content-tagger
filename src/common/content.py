from dataclasses import dataclass
from typing import Any
import base64
import dacite
import json
import requests

from elv_client_py import ElvClient

from src.common.errors import BadRequestError, ExternalServiceError
from src.common.logging import logger

@dataclass(frozen=True)
class Content:
    qid: str
    token: str

@dataclass
class ContentConfig:
    config_url: str
    parts_url: str
    live_media_url: str

class QAPI:
    """Content object representation and API wrapper.

    Serves the following uses:
        - Caches qinfo call (qid, qhash, qlib) so that they don't need to be requeried
        - Bridges the the parts and regular fabric APIs into one interface
        - Helps reduce parameter counts in functions by not requiring the qid & the ElvClient to
            be passed separately
    """

    def __init__(
        self, 
        q: Content,
        cfg: ContentConfig
    ):
        try:
            client = ElvClient.from_configuration_url(
                cfg.config_url, static_token=q.token)
        except Exception as e:
            raise ExternalServiceError("Failed to create content client") from e
        
        try:
            parts_client = ElvClient.from_configuration_url(
                cfg.parts_url, static_token=q.token)
        except Exception as e:
            logger.opt(exception=e).error("Failed to create parts client")
            parts_client = None
        
        try:
            live_client = ElvClient.from_configuration_url(
                cfg.live_media_url, static_token=q.token)
        except Exception as e:
            logger.opt(exception=e).error("Failed to create live media client")
            live_client = None

        # will raise HTTPError if auth is invalid or qid is not found
        qinfo = client.content_object(**parse_qhit(q.qid))

        self.qid = qinfo["id"]
        self.qhash = qinfo.get("hash", None)
        self.qwt = qinfo.get("write_token", None)

        assert self.qhash or self.qwt, f"Content object must have either a hash or a write token. {qinfo}"

        self.qlib = qinfo["qlib_id"]
        self.qid = q.qid
        self._client = client
        self._parts_client = parts_client
        self._live_client = live_client

        self.cfg = cfg

    def id(self) -> str:
        return self.qid

    def token(self) -> str:
        return self._client.token

    def content_object_versions(self) -> dict[str, Any]:
        """Get all versions of the content object."""
        return self._client.content_object_versions(object_id=self.qid, library_id=self.qlib)
    
    def download_part(self, **kwargs) -> None:
        """Download a part of the content object."""
        if self._parts_client is None:
            raise RuntimeError("Parts client is not initialized")
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
        if self._live_client is None:
            raise RuntimeError("Live media client is not initialized")
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
        return f"Content(qid={self.qid}, qid={self.qid}, qhash={self.qhash}, qlib={self.qlib})"

class QAPIFactory:
    def __init__(self, cfg: ContentConfig):
        self.cfg = cfg

    def create(self, q: Content) -> QAPI:
        return QAPI(q, self.cfg)

def parse_qhit(qid: str) -> dict[str, str]:
    """Parse a qid into a dictionary so it can be passed to elv_client_py functions 
    and use the correct argument."""
    if not isinstance(qid, str):
        raise BadRequestError(f"qid must be a string, got {type(qid)}")

    if qid.startswith("hq__"):
        return {"version_hash": qid}
    elif qid.startswith("tqw__"):
        return {"write_token": qid}
    elif qid.startswith("iq__"):
        return {"object_id": qid}

    raise BadRequestError(f"Invalid qid: {qid}")

