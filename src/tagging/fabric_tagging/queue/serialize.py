from dataclasses import asdict

from dacite import from_dict

from src.fetch.model import *
from src.service.model import TagDetails
from src.tagging.fabric_tagging.model import TagArgs
from src.tagging.fabric_tagging.queue.dto import TagDetailsRaw

def _convert_scope(data: dict) -> Scope:
    type = data.get("type")
    if type == "processor":
        return TimeRangeScope(**data)
    elif type == "assets":
        return AssetScope(**data)
    elif type == "video":
        return VideoScope(**data)
    elif type == "livestream":
        return LiveScope(**data)
    elif type == "tag-aligned":
        return TagAlignedScope(**data)
    else:
        raise ValueError(f"Unknown scope type: {type}")

def params_to_dict(params: TagArgs) -> dict:
    return asdict(params)

def params_from_dict(p: dict | None) -> TagArgs | None:
    if not p:
        return None
    return TagArgs(
        feature=p["feature"],
        run_config=p["run_config"],
        scope=_convert_scope(p["scope"]),
        track_suffix=p.get("track_suffix", ""),
        replace=p["replace"],
        destination_qid=p["destination_qid"],
        index_qid=p.get("index_qid", ""),
        max_fetch_retries=p["max_fetch_retries"],
        caller_info=p.get("caller_info", {})
    )

def details_to_dict(details: TagDetails) -> dict:
    return asdict(details)

def details_from_dict(d: dict | None) -> TagDetails | None:
    # the queue manager writes status_details holding only a system_message, e.g. on cascaded cancels
    if not d or "tag_status" not in d:
        return None
    return from_dict(TagDetailsRaw, d).to_model()

def system_message(d: dict | None) -> str | None:
    if not d:
        return None
    return d.get("system_message")
