import argparse
from typing import List, Optional, Literal, Dict, Iterable, Tuple
from flask import Flask, request, Response, Request
from flask_cors import CORS
from podman import PodmanClient
import json
from loguru import logger
from dataclasses import dataclass, asdict, field
import os
from elv_client_py import ElvClient
from queue import Queue
from collections import defaultdict
import threading
from requests.exceptions import HTTPError
import time
import shutil
import signal
import atexit
from marshmallow import ValidationError, fields, Schema
from common_ml.types import Data
from common_ml.utils.metrics import timeit

from config import config
from src.fabric.utils import parse_qhit
from src.fabric.agg import format_video_tags, format_asset_tags
from src.fabric.video import download_stream, StreamNotFoundError
from src.fabric.assets import fetch_assets, AssetsNotFoundException

from src.manager import ResourceManager, NoGPUAvailable


client = ElvClient.from_configuration_url(config_url="https://host-76-74-28-229.contentfabric.io/config?self&qspace=main", static_token=os.environ["TOKEN"])


content_args = parse_qhit("iq__4Hshtz5JeH9GxDBvxNrtPWgtUfZL")
qlib = client.content_object_library_id(**content_args)

os.makedirs("here", exist_ok=True)
res = client.download_directory(dest_path="here", fabric_path=f"video_tags/video/shot", **content_args)

print(res)



    
