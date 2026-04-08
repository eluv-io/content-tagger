import pytest
import json
import os
from unittest.mock import Mock, patch

from src.tag_containers.model import ModelTag

def test_model_tag_hashing():
    tag1 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr")
    tag2 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr")
    tag3 = ModelTag(start_time=1000, end_time=6000, text="world", source_media="media2.mp4", model_track="caption")
    
    assert hash(tag1) == hash(tag2), "Identical tags should have the same hash"
    assert hash(tag1) != hash(tag3), "Different tags should have different hashes"

def test_model_tag_equality():
    tag1 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr")
    tag2 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr")
    tag3 = ModelTag(start_time=1000, end_time=6000, text="world", source_media="media2.mp4", model_track="caption")
    
    assert tag1 == tag2, "Identical tags should be equal"
    assert tag1 != tag3, "Different tags should not be equal"

def test_model_tag_hashing_frameinfo():
    tag1 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr", frame_info={"frame_idx": 10})
    tag2 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr", frame_info={"frame_idx": 10})
    tag3 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr", frame_info={"frame_idx": 10, "box": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}})
    
    assert hash(tag1) == hash(tag2), "Identical tags should have the same hash"
    assert hash(tag1) != hash(tag3), "Different tags should have different hashes"    

def test_model_tag_equality_frameinfo():
    tag1 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr", frame_info={"frame_idx": 10})
    tag2 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr", frame_info={"frame_idx": 10})
    tag3 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr", frame_info={"frame_idx": 10, "box": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}})
    
    assert tag1 == tag2, "Identical tags should be equal"
    assert tag1 != tag3, "Different tags should not be equal"

def test_model_tag_complex_hashing():
    tag1 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr",
                    additional_info={"key": "value", "key2": "value2"})
    tag2 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr",
                    additional_info={"key2": "value2", "key": "value"})

    assert hash(tag1) == hash(tag2), "Tags with identical additional_info should have the same hash"

    tag3 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr",
                    additional_info={"key2": "value", "key": [1, 2, "arbitrary", {"hey": "a_dict"}]})
    tag4 = ModelTag(start_time=0, end_time=5000, text="hello", source_media="media1.mp4", model_track="asr",
                    additional_info={"key": [1, 2, "arbitrary", {"hey": "a_dict"}], "key2": "value"})

    assert hash(tag3) == hash(tag4), "Tags with identical additional_info should have the same hash"
    
    assert hash(tag1) != hash(tag3), "Tags with different additional_info should have different hash"

    


