"""
These functions map between the API DTOs and service layer structs.
"""

from flask import request
from requests import HTTPError
from dacite import from_dict, Config

from common_ml.utils.dictionary import nested_update

from src.api.tagging.request_format import *
from src.fetch.model import *
from src.tag_containers.registry import ContainerRegistry
from src.tagging.fabric_tagging.model import TagArgs
from src.common.content import Content
from src.common.errors import BadRequestError, MissingResourceError

def map_video_tag_dto(
    args: StartJobsRequest, 
    registry: ContainerRegistry,
    q: Content
) -> list[TagArgs]:
    """
    Map video tagging API arguments to internal TagArgs structures.
    """
    defaults = args.options
    if len(args.jobs) == 0:
        raise BadRequestError("Please specify at least one job to run.")
    res = []
    for job in args.jobs:
        tag_arg = _set_defaults(q, defaults, job, registry)
        res.append(tag_arg)
    return res

def _set_defaults(
    q: Content,
    defaults: TaggerOptions,
    job: JobSpec,
    registry: ContainerRegistry
) -> TagArgs:
    feature = job.model
    run_config = job.model_params
    overrides = job.overrides or TaggerOptions()

    destination_qid = overrides.destination_qid or defaults.destination_qid
    replace = overrides.replace or defaults.replace
    max_fetch_retries = overrides.max_fetch_retries

    is_live = is_live_content(q)

    default_scope = _get_default_scope_dict(is_live, registry.get_model_config(feature).type, q)

    # override with options provided in request
    scope_dict = nested_update(default_scope, defaults.scope)
    # override with per-model options provided in request
    scope_dict = nested_update(scope_dict, overrides.scope)

    try:
        scope = _map_scope(scope_dict)
    except Exception as e:
        raise BadRequestError(f"Invalid scope configuration: {e}") from e

    return TagArgs(
        feature=feature,
        run_config=run_config,
        scope=_scope_dto_to_model(scope),
        replace=replace,
        destination_qid=destination_qid,
        max_fetch_retries=max_fetch_retries,
    )

def _map_scope(scope_arg: dict[str, Any]) -> ScopeDTO:
    scope_type = scope_arg.get("type")    
    if scope_type == "video":
        return ScopeVideo(**scope_arg)
    elif scope_type == "processor":
        return ScopeProcessor(**scope_arg)
    elif scope_type == "assets":
        del scope_arg["stream"]
        return ScopeAssets(**scope_arg)
    elif scope_type == "livestream":
        return ScopeLivestream(**scope_arg)
    else:
        raise BadRequestError(f"Invalid scope type: {scope_type}")
    
def _scope_dto_to_model(scope: ScopeDTO) -> Scope:
    if isinstance(scope, ScopeVideo):
        return VideoScope(
            stream=scope.stream,
            start_time=scope.start_time,
            end_time=scope.end_time,
        )
    elif isinstance(scope, ScopeProcessor):
        return TimeRangeScope(
            stream=scope.stream,
            start_time=scope.start_time,
            end_time=scope.end_time,
            chunk_size=scope.chunk_size,
        )
    elif isinstance(scope, ScopeAssets):
        return AssetScope(assets=scope.assets)
    elif isinstance(scope, ScopeLivestream):
        return LiveScope(
            chunk_size=scope.segment_length,
            max_duration=scope.max_duration,
            stream=scope.stream,
        )
    else:
        raise BadRequestError(f"Invalid scope type: {type(scope)}")

def _get_default_scope_dict(is_live: bool, model_type: str, q: Content) -> dict[str, Any]:
    res = {}
    if is_live and model_type == "processor":
        raise BadRequestError("Processor models are not currently supported for live content.")

    if is_live:
        res["type"] = "livestream"
    elif model_type == "processor":
        res["type"] = "processor"
    else:
        res["type"] = "video"

    if model_type == "audio" and not is_live:
        res["stream"] = _find_default_audio_stream(q)
    else:
        res["stream"] = "video"

    return res

def _find_default_audio_stream(q: Content) -> str:
    streams = q.content_object_metadata(
        metadata_subtree="offerings/default/media_struct/streams",
        resolve_links=False,
    )

    assert isinstance(streams, dict)

    # First pass: filter to only audio streams
    audio_streams = {
        name: info for name, info in streams.items()
        if info.get("codec_type") == "audio"
    }
    
    if not audio_streams:
        raise MissingResourceError("No audio streams found")
    
    for stream_name, stream_info in audio_streams.items():
        if stream_info.get("language") == "en" and stream_info.get("channels") == 2:
            return stream_name
    
    for stream_name, stream_info in audio_streams.items():
        if stream_info.get("language") == "en":
            return stream_name
    
    for stream_name, stream_info in audio_streams.items():
        if stream_info.get("channels") == 2:
            return stream_name
    
    return list(audio_streams.keys())[0]

def is_live_content(q: Content) -> bool:
    try:
        edge_write_token = q.content_object_metadata(
            metadata_subtree="live_recording/status/edge_write_token",
            resolve_links=False,
        )
    except HTTPError:
        return False

    return isinstance(edge_write_token, str) and edge_write_token.startswith("tqw__")