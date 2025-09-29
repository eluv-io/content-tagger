from flask import jsonify, request, Response, current_app
import dacite

from src.api.tagging.handlers import _get_authorized_content
from src.api.upload.format import FinalizeArgs
from src.common.errors import BadRequestError
from src.tags.conversion import TagConverter
from src.tags.conversion_workflow import upload_tags_to_fabric
from src.tags.tagstore.abstract import Tagstore

def handle_commit(qhit: str) -> Response:

    qsource = _get_authorized_content(qhit)

    try:
        args = dacite.from_dict(data=request.args, data_class=FinalizeArgs)
    except Exception as e:
        raise BadRequestError(f"Invalid request: {str(e)}")
    
    qwt = _get_authorized_content(args.write_token)

    tag_converter: TagConverter = current_app.config["state"]["tag_converter"]
    tagstore: Tagstore = current_app.config["state"]["tagger"].tagstore

    upload_tags_to_fabric(
        source_q=qsource,
        qwt=qwt,
        tag_converter=tag_converter,
        tagstore=tagstore
    )

    qwt.set_commit_message(message="Uploaded tags via tagger worker")

    return jsonify({"status": "finalized"})