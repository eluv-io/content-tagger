
from functools import lru_cache
from copy import deepcopy

from common_ml.utils.dictionary import nested_update
from requests import HTTPError

from src.common.content import QAPI, Content, QAPIFactory
from src.common.model import ModelConfig
from src.fetch.model import *
from src.tag_containers.registry import ContainerRegistry
from src.api.tagging.request_format import *
from src.tagging.fabric_tagging.model import TagArgs, Scope
from src.common.errors import BadRequestError, MissingResourceError

class ArgsResolver:
    """Class to resolve arguments for tagging features."""

    def __init__(self, model_configs: dict[str, ModelConfig], api_factory: QAPIFactory):
        self.model_configs = model_configs
        self.api_factory = api_factory

    def resolve(self, args: StartJobsRequest, q: Content) -> list[TagArgs]:
        """
        Resolve API arguments to internal TagArgs structures.
        """
        defaults = args.options
        if len(args.jobs) == 0:
            raise BadRequestError("Please specify at least one job to run.")
        res = []
        for job in args.jobs:
            tag_arg = self._set_defaults(q, defaults, job)
            res.append(tag_arg)
        return res
    
    @lru_cache(maxsize=1024)
    def find_default_audio_stream(self, q: Content) -> str:
        qapi = self.api_factory.create(q)
        streams = qapi.content_object_metadata(
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

    @lru_cache(maxsize=1024)
    def is_live_content(self, q: Content) -> bool:
        qapi = self.api_factory.create(q)
        try:
            edge_write_token = qapi.content_object_metadata(
                metadata_subtree="live_recording/status/edge_write_token",
                resolve_links=False,
            )
        except HTTPError:
            return False

        return isinstance(edge_write_token, str) and edge_write_token.startswith("tqw__")

    def _set_defaults(
        self,
        q: Content,
        defaults: TaggerOptions,
        job: JobSpec,
    ) -> TagArgs:
        feature = job.model
        run_config = job.model_params
        overrides = job.overrides

        destination_qid = overrides.destination_qid if overrides.destination_qid \
            is not None else defaults.destination_qid
        replace = overrides.replace if overrides.replace \
            is not None else defaults.replace
        max_fetch_retries = overrides.max_fetch_retries if overrides.max_fetch_retries \
            is not None else defaults.max_fetch_retries

        # set defaults for options that are not provided in request
        if destination_qid is None:
            destination_qid = ""
        if replace is None:
            replace = False
        if max_fetch_retries is None:
            max_fetch_retries = 3

        model_cfg = self.model_configs.get(feature)
        if model_cfg is None:
            raise BadRequestError(f"Model {feature} not found.")
        
        model_type = model_cfg.type

        default_scope = self._get_default_scope_dict(model_type, model_cfg.scope or {}, q)
        
        # override with options provided in request
        scope_dict = nested_update(default_scope, defaults.scope)
        # override with per-model options provided in request
        scope_dict = nested_update(scope_dict, overrides.scope)
        
        try:
            scope = self._map_scope(scope_dict)
        except Exception as e:
            raise BadRequestError(f"Invalid scope configuration: {e}") from e

        return TagArgs(
            feature=feature,
            run_config=run_config,
            scope=scope,
            replace=replace,
            track_suffix=job.track_suffix,
            destination_qid=destination_qid,
            max_fetch_retries=max_fetch_retries,
            caller_info=job.caller_info
        )

    def _map_scope(self, scope_arg: dict[str, Any]) -> Scope:
        scope_type = scope_arg.get("type")    
        if scope_type == "video":
            return VideoScope(**scope_arg)
        elif scope_type == "processor":
            return TimeRangeScope(**scope_arg)
        elif scope_type == "assets":
            del scope_arg["stream"]
            return AssetScope(**scope_arg)
        elif scope_type == "livestream":
            return LiveScope(**scope_arg)
        elif scope_type == "tag-aligned":
            return TagAlignedScope(**scope_arg)
        else:
            raise BadRequestError(f"Invalid scope type: {scope_type}")

    def _get_default_scope_dict(self, model_type: str, model_scope: dict, q: Content) -> dict[str, Any]:
        res = {}
        is_live = self.is_live_content(q)
        if is_live and model_type == "processor":
            raise BadRequestError("Processor models are not currently supported for live content.")

        res = deepcopy(model_scope)

        # set type
        if "type" not in res:
            if is_live:
                res["type"] = "livestream"
            elif model_type == "processor":
                res["type"] = "processor"
            else:
                res["type"] = "video"

        if "stream" not in res:
            if model_type == "audio" and not is_live:
                res["stream"] = self.find_default_audio_stream(q)
            else:
                res["stream"] = "video"

        return res