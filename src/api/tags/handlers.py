
import os
import json
import shutil
import tempfile
from flask import Response, request, current_app
from loguru import logger
from requests.exceptions import HTTPError

from elv_client_py import ElvClient
from common_ml.utils.metrics import timeit

from src.api.tags.format import FinalizeArgs, UploadArgs
from src.api.errors import BadRequestError, MissingResourceError
from src.api.auth import authenticate, parse_qhit, get_client
from src.tagger.tagger import Tagger
from src.fabric.content import Content
from src.fabric.agg import format_video_tags, format_asset_tags

from config import config

def handle_finalize(qhit: str) -> Response:
    finalize_lock = current_app.config["state"]["finalize_lock"]
    try:
        args = FinalizeArgs.from_dict(request.args)
    except Exception as e:
        raise BadRequestError(f"Invalid request: {str(e)}")
    with finalize_lock[args.write_token]:
        return _finalize_internal(qhit, args, True)
    
def handle_aggregate(qhit: str) -> Response:
    finalize_lock = current_app.config["state"]["finalize_lock"]
    try:
        args = FinalizeArgs.from_dict(request.args)
    except Exception as e:
        raise BadRequestError(f"Invalid request: {str(e)}")
    with finalize_lock[args.write_token]:
        return _finalize_internal(qhit, args, False)
    
def handle_upload(qhit: str) -> Response:
    uploaded_files = request.files.getlist('file')
    if len(uploaded_files) == 0:
        raise BadRequestError("No files in request")
    
    try:
        args = UploadArgs.from_dict(request.args)
    except Exception as e:
        raise BadRequestError(f"Invalid request: {str(e)}")

    to_upload = []
    for file in uploaded_files:
        try:
            filedata = json.load(file.stream)
        except json.JSONDecodeError:
            return Response(response=json.dumps({'error': 'Invalid JSON file'}), status=400, mimetype='application/json')
        to_upload.append((file.filename, filedata))

    filesystem_lock = current_app.config["state"]["filesystem_lock"]

    with filesystem_lock:
        os.makedirs(os.path.join(config["storage"]["tags"], qhit, 'external_tags'), exist_ok=True)

        for fname, fdata in to_upload:
            if os.path.exists(os.path.join(config["storage"]["tags"], qhit, 'external_tags', fname)):
                logger.warning(f"File {fname} already exists, overwriting")
            with open(os.path.join(config["storage"]["tags"], qhit, 'external_tags', fname), 'w') as f:
                json.dump(fdata, f)
                
    if not args.write_token:
        args.write_token = qhit

    if args.aggregate:
        return _finalize_internal(qhit, args, True)

    return Response(response=json.dumps({'message': 'Successfully uploaded tags'}), status=200, mimetype='application/json')

def _finalize_internal(qhit: str, args: FinalizeArgs, upload_local_tags = True) -> Response:
    qwt = args.write_token
    client = get_client(request, qwt, config["fabric"]["config_url"])

    if not authenticate(client, qhit):
        # make sure that the auth token has access to the content object where the tags are from
        # TODO: we may need to check more permissions to make sure the user should be able to read the tags. 
        return Response(response=json.dumps({'error': 'Unauthorized'}), status=403, mimetype='application/json')
    
    content_args = parse_qhit(qwt)
    qlib = client.content_object_library_id(**content_args)

    q = Content(qhit, args.authorization)

    filesystem_lock = current_app.config["state"]["filesystem_lock"]

    tagger: Tagger = current_app.config["state"]["tagger"]

    file_jobs = []
    if upload_local_tags:
        running_jobs = tagger.get_running_jobs(qhit)
        
        if len(running_jobs) > 0 and not args.force:
            raise BadRequestError(
                "Some jobs are still running. Use `force=true` to finalize anyway."
            )

        if not os.path.exists(os.path.join(config["storage"]["tags"], qhit)):
            raise MissingResourceError(
                f"No tags found for {qhit}."
            )

        for stream in os.listdir(os.path.join(config["storage"]["tags"], qhit)):
            if stream == "external_tags":
                continue
            for feature in os.listdir(os.path.join(config["storage"]["tags"], qhit, stream)):
                tagged_media_files = []
                for tag in os.listdir(os.path.join(config["storage"]["tags"], qhit, stream, feature)):
                    tagfile = os.path.join(config["storage"]["tags"], qhit, stream, feature, tag)
                    tagged_media_files.append(tagger._source_from_tag_file(tagfile))
                tagged_media_files = list(set(tagged_media_files))
                num_files = len(tagged_media_files)
                if not args.replace:
                    with timeit(f"Filtering tagged files for {qhit}, {feature}, {stream}"):
                        tagged_media_files = tagger._filter_tagged_files(tagged_media_files, q, qhit, stream, feature)
                logger.debug(f"Upload status for {qhit}: {feature} on {stream}\nTotal media files: {num_files}, Media files to upload: {len(tagged_media_files)}, Media files already uploaded: {num_files - len(tagged_media_files)}")
                if not tagged_media_files:
                    continue
                if stream == "image":
                    for source in tagged_media_files:
                        tagfile = source + "_imagetags.json"
                        if not os.path.exists(tagfile):
                            logger.warning(f"Expected tag file {tagfile} not found, skipping")
                            continue
                        file_jobs.append(ElvClient.FileJob(local_path=tagfile,
                                                out_path=f"image_tags/{feature}/{os.path.basename(tagfile)}",
                                                mime_type="application/json"))
                else:
                    for source in tagged_media_files:
                        tagfile = source + "_tags.json"
                        if not os.path.exists(tagfile):
                            # this should only happen if force=True and frametags get written before video tags
                            logger.warning(f"Expected tag file {tagfile} not found, skipping.")
                            continue
                        file_jobs.append(ElvClient.FileJob(local_path=tagfile,
                                                out_path=f"video_tags/{stream}/{feature}/{os.path.basename(tagfile)}",
                                                mime_type="application/json"))

                        if os.path.exists(source + "_frametags.json"):
                            file_jobs.append(ElvClient.FileJob(local_path=source + "_frametags.json",
                                                out_path=f"video_tags/{stream}/{feature}/{os.path.basename(source)}_frametags.json",
                                                mime_type="application/json"))
                            
        external_tags_path = os.path.join(config["storage"]["tags"], qhit, "external_tags")
        if os.path.exists(external_tags_path):
            local_source_tags = os.listdir(external_tags_path)
            try:
                remote_source_tags = client.list_files(qlib, path="video_tags/source_tags/user", **content_args)
            except HTTPError:
                logger.debug(f"No source tags found for {qwt}")
                remote_source_tags = []
            
            for local_source in local_source_tags:
                tagfile = os.path.join(config["storage"]["tags"], qhit, "external_tags", local_source)
                file_jobs.append(ElvClient.FileJob(local_path=tagfile,
                                                    out_path=f"video_tags/source_tags/user/{local_source}",
                                                    mime_type="application/json"))
                # TODO: we do need a way to make it so we can do replace=true for external tags but not on the rest. if we can improve the efficiency of this step we could just do two passes and 
                # Let the user specify which features they want to finalize, and they could do two steps. For now, we will default to always overwriting the external tags.
                if local_source in remote_source_tags:
                    logger.warning(f"External tag file {local_source} already exists, overwriting")

    if len(file_jobs) > 0:
        try:
            logger.debug(f"Uploading {len(file_jobs)} tag files")
            with timeit("Uploading tag files"):
                client.upload_files(library_id=qlib, file_jobs=file_jobs, finalize=False, **content_args)
        except HTTPError as e:
            return Response(json.dumps({'error': str(e), 'message': 'Please verify your authorization token has write access and the write token has not already been committed. This error can also arise if the write token has already been used to finalize tags.'}), status=403, mimetype='application/json')
        except ValueError as e:
            return Response(response=json.dumps({'error': str(e), 'message': 'Please verify the provided write token has not already been used to finalize tags.'}), status=400, mimetype='application/json')
    # if no file jobs, then we just do the aggregation

    tmpdir = tempfile.TemporaryDirectory(dir=config["storage"]["tmp"])

    with filesystem_lock:
        if os.path.exists(os.path.join(config["storage"]["tags"], qhit)):
            shutil.copytree(os.path.join(config["storage"]["tags"], qhit), tmpdir.name, dirs_exist_ok=True)
        if os.path.exists(os.path.join(tmpdir.name, 'external_tags')):
            shutil.rmtree(os.path.join(tmpdir.name, 'external_tags'))

    try:
        with timeit("Aggregating video tags"):
            format_video_tags(client, qwt, config["agg"]["interval"], tmpdir.name)
        with timeit("Aggregating asset tags"):
            format_asset_tags(client, qwt, tmpdir.name)
    except HTTPError as e:
        message = (
            "Please verify your authorization token has write access and the write token has not already been committed."
            "This error can also arise if the write token has already been used to finalize tags."
        )
        return Response(response=json.dumps({'error': str(e), 'message': message}), status=403, mimetype='application/json')
    finally:
        if "keeplasttemp" in os.environ.get("TAGGER_AGG", ""):
            ## for debugging, keep the last temp directory
            global last_tmpdir
            last_tmpdir = tmpdir
        else:
            tmpdir.cleanup()

    if not args.leave_open:
        client.finalize_files(qwt, qlib)

    client.set_commit_message(qwt, "uploaded/aggregated ML tags (taggerv2)", qlib)

    return Response(response=json.dumps({'message': 'Succesfully uploaded tag files. Please finalize the write token.', 'write token': qwt}), status=200, mimetype='application/json')