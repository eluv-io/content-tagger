"""
These functions map between the API DTOs and service layer structs.
"""

from src.api.tagging.format import *
from src.fetch.model import *
from src.tag_containers.registry import ContainerRegistry
from src.tagging.fabric_tagging.model import TagArgs
from src.common.content import Content

def map_video_tag_dto(
        args: TagAPIArgs, 
        registry: ContainerRegistry,
        q: Content
) -> list[TagArgs]:
    """
    Map video tagging API arguments to internal TagArgs structures.
    """
    res = []
    for feature, config in args.features.items():
        if config.stream is not None:
            stream = config.stream
        else:
            model_config = registry.get_model_config(feature)
            model_type = model_config.type
            if model_type in ("video", "frame"):
                stream = "video"
            else:
                stream = _find_default_audio_stream(q)
            config.stream = stream

        start_time = args.start_time
        if args.start_time is None:
            start_time = 0
        # help the type checker
        assert isinstance(start_time, int)

        end_time = args.end_time
        if args.end_time is None:
            end_time = float('inf')
        assert isinstance(end_time, int) or isinstance(end_time, float)

        res.append(TagArgs(feature=feature, run_config=config.run_config, scope=VideoScope(config.stream, start_time=start_time, end_time=end_time), replace=args.replace))

    return res

def map_asset_tag_dto(args: ImageTagAPIArgs) -> list[TagArgs]:
    res = []
    for feature, config in args.features.items():
        res.append(TagArgs(feature=feature, run_config=config.run_config, scope=AssetScope(assets=args.assets), replace=args.replace))
        
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