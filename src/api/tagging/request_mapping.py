"""
These functions map between the API DTOs and service layer structs.
"""

from flask import request
from requests import HTTPError

from common_ml.utils.dictionary import nested_update

from src.api.tagging.request_format import *
from src.fetch.model import *
from src.service.model import StatusArgs
from src.tag_containers.registry import ContainerRegistry
from src.tagging.fabric_tagging.model import TagArgs
from src.common.content import Content
from src.common.errors import BadRequestError, MissingResourceError

def status_request_to_internal(req: StatusRequest) -> StatusArgs:
    return StatusArgs(
        qid=req.qid,
        user=req.user,
        tenant=req.tenant,
        title=req.title
    )