from flask import Response, request, current_app

from src.api.tagging.format import TagArgs
from src.api.errors import BadRequestError, MissingResourceError
from src.api.auth import get_authorization
from src.fabric.content import Content
from src.tagger.tagger import Tagger

def handle_tag(qhit: str) -> Response:
    auth = get_authorization(request)
    
    q = Content(qhit, auth)

    try:
        args = TagArgs.from_dict(request.json)
    except BadRequestError as e:
        raise e
    except Exception as e:
        raise BadRequestError(
            f"Invalid search arguments. Please check your query parameters") from e

    tagger: Tagger = current_app.config["state"]["tagger"]
