from dataclasses import dataclass, asdict
import json

import requests

from config import config

@dataclass
class TagRow:
    # Represents a row in the video_tags table
    start_time: float
    end_time: float
    content: str
    source: str
    track: str
    additional_info: dict

def upload_tags(qid: str, tags: list[TagRow]) -> None:
    tags = [asdict(tag).pop("qid") for tag in tags]
    jsonbody = {"qid": tags}
    requests.post(f"{config['api']['url']}/tags", json=jsonbody)

def load_tag_file(
        feature: str, 
        filepath: str
    ) -> list[TagRow]:
    # Load part tag file to list of TagRow

    with open(filepath, "r") as f:
        parttags = json.load(f)

    return [TagRow(
        start_time=int(tagdata["start_time"]),
        end_time=int(tagdata["end_time"]),
        content=tagdata["text"],
        source="Tagger",
        track=feature,
        additional_info={}
    ) for tagdata in parttags]

def check_qid(qhit: str) -> bool:
    return qhit.startswith("iq__")