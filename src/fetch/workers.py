import threading
import os
import tempfile
from loguru import logger
import shutil
from copy import deepcopy
from contextlib import contextmanager
import math
import json

from common_ml.video_processing import unfrag_video

from requests import HTTPError

from src.fetch.model import *
from src.common.content import Content, ContentConfig
from src.common.errors import BadRequestError, MissingResourceError
from src.fetch.model import DownloadResult
from src.fetch.video_process import center_segment