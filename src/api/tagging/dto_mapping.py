"""
These functions map between the API DTOs and service layer structs.
"""

from flask import request
from requests import HTTPError
from dacite import from_dict

from src.api.tagging.format import *
from src.fetch.model import *
from src.tag_containers.registry import ContainerRegistry
from src.tagging.fabric_tagging.model import TagArgs
from src.common.content import Content
from src.common.errors import BadRequestError

def map_video_tag_dto(
        args: TagAPIArgs | LiveTagAPIArgs, 
        registry: ContainerRegistry,
        q: Content
) -> list[TagArgs]:
    """
    Map video tagging API arguments to internal TagArgs structures.
    """
    if isinstance(args, LiveTagAPIArgs):
        return map_live_tag_dto(args, registry)
    else:
        return map_vod_tag_dto(args, registry, q)
    
def _create_tag_args(
    feature: str,
    config: ModelParams,
    scope: Scope,
    args: BaseTagAPIArgs
) -> TagArgs:
    """Create TagArgs - single place to extract common fields from args."""
    return TagArgs(
        feature=feature,
        run_config=config.model,
        scope=scope,
        replace=args.replace,
        destination_qid=args.destination_qid
    )

def map_vod_tag_dto(args: TagAPIArgs, registry: ContainerRegistry, q: Content) -> list[TagArgs]:
    res = []
    for feature, config in args.features.items():
        if config.stream is None:
            model_config = registry.get_model_config(feature)
            model_type = model_config.type
            if model_type in ("video", "frame"):
                stream = "video"
            else:
                stream = _find_default_audio_stream(q)
            config.stream = stream

        start_time = args.start_time if args.start_time is not None else 0
        end_time = args.end_time if args.end_time is not None else float('inf')

        res.append(_create_tag_args(
            feature=feature,
            config=config,
            scope=VideoScope(config.stream, start_time=start_time, end_time=end_time),
            args=args,
        ))
    return res

def map_live_tag_dto(args: LiveTagAPIArgs, registry: ContainerRegistry) -> list[TagArgs]:
    res = []
    for feature, config in args.features.items():
        model_config = registry.get_model_config(feature)
        if model_config.type == "audio":
            raise BadRequestError(f"Feature {feature} is of type audio, but live tagging only supports video models")
        
        res.append(_create_tag_args(
            feature=feature,
            config=config,
            scope=LiveScope("video", chunk_size=args.segment_length, max_duration=args.max_duration),
            args=args,
        ))
    return res

def map_asset_tag_dto(args: ImageTagAPIArgs) -> list[TagArgs]:
    res = []
    for feature, config in args.features.items():
        res.append(_create_tag_args(
            feature=feature,
            config=config,
            scope=AssetScope(assets=args.assets),
            args=args
        ))
    return res

def _find_default_audio_stream(q: Content) -> str:
    # TODO: will this work for live?
    
    streams = q.content_object_metadata(
        metadata_subtree="offerings/default/media_struct/streams",
        resolve_links=False,
    )

    assert isinstance(streams, dict)

    for stream_name, stream_info in streams.items():
        if stream_info.get("codec_type") == "audio" and \
            stream_info.get("language") == "en" and \
            stream_info.get("channels") == 2:
            return stream_name
        
    return "audio"

def tag_args_from_req(q: Content) -> TagAPIArgs | LiveTagAPIArgs:
    try:
        body = request.json
        assert body is not None
        if _is_live(q):
            args = from_dict(LiveTagAPIArgs, body)
        else:
            args = from_dict(TagAPIArgs, body)
    except Exception as e:
        raise BadRequestError(f"Invalid request body: {e}") from e
    
    return args
    
def _is_live(q: Content) -> bool:
    try:
        edge_write_token = q.content_object_metadata(
            metadata_subtree="live_recording/status/edge_write_token",
            resolve_links=False,
        )
    except HTTPError:
        return False

    return isinstance(edge_write_token, str) and edge_write_token.startswith("tqw__")