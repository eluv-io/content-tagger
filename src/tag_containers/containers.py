from copy import deepcopy
import uuid
from podman import PodmanClient
from src.common.logging import logger
import json
import os
from datetime import datetime
from copy import copy

from common_ml.utils.files import get_file_type
from common_ml.video_processing import get_fps

from src.tag_containers.model import FrameTag
from src.common.errors import MissingResourceError, BadRequestError
from src.tags.tagstore.types import Tag
from src.tag_containers.model import *

class TagContainer:

    def __init__(
        self,
        pclient: PodmanClient,
        cfg: ContainerSpec
    ):
        self.cfg = cfg

        if isinstance(self.cfg.media_input, str):
            # single directory
            if not os.path.exists(self.cfg.media_input):
                raise FileNotFoundError(f"Directory {self.cfg.media_input} not found")
            elif not os.path.isdir(self.cfg.media_input):
                raise NotADirectoryError(f"{self.cfg.media_input} is not a directory")
            self.media_files = [os.path.join(self.cfg.media_input, f) for f in os.listdir(self.cfg.media_input)]
        else:
            self.media_files = self.cfg.media_input

        self.media_dir = self._find_common_root(self.media_files)

        max_depth = self._calculate_max_depth(self.media_files, self.media_dir)
        
        if max_depth > 3:
            raise BadRequestError(f"Files are too deeply nested ({max_depth} levels below common root). Maximum allowed depth is 3.\n {self.media_files}")

        file_types = [get_file_type(f) for f in self.media_files]
        if len(set(file_types)) > 1:
            raise BadRequestError(f"All files must be of the same type: {self.media_files}")
        if len(file_types) == 0:
            raise ValueError("No files provided")
        self.file_type = file_types[0]
        if self.file_type not in ["video", "audio", "image"]:
            raise BadRequestError(f"Unsupported file type: {self.file_type}")
        # check that no file has the same basename
        self.basename_to_source = {os.path.basename(f): f for f in self.media_files}
        if len(self.basename_to_source) != len(self.media_files):
            raise BadRequestError("Files must have unique basenames")
        self.pclient = pclient
        self.container = None

    def start(
        self, 
        gpuidx: int | None,
    ) -> None:
        os.makedirs(self.cfg.tags_dir, exist_ok=True)
        
        volumes = [
            {
                "source": self.cfg.tags_dir,
                # convention for containers to store tags in /elv/tags
                "target": "/elv/tags",
                "type": "bind",
            },
            {
                "source": self.cfg.cache_dir,
                # convention for python modules to store cache in /root/.cache
                "target": "/root/.cache",
                "type": "bind",
                "read_only": False
            },
            {
                "source": self.media_dir,
                "target": "/elv/media",
                "type": "bind",
                "read_only": True
            }
        ]

        # Calculate paths from perspective of the container working directory "/elv"
        relative_paths = []
        for f in self.media_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"File {f} not found")
            elif not os.path.isfile(f):
                raise IsADirectoryError(f"{f} is a directory")
            elif not os.path.isabs(f):
                raise ValueError(f"{f} must be an absolute path")
            
            rel_path = os.path.relpath(f, self.media_dir)
            relative_paths.append(f"media/{rel_path}")

        kwargs = {
            "image": self.cfg.model_config.image,
            "command": relative_paths + ["--config", f"{json.dumps(self.cfg.run_config)}"],
            "mounts": volumes,
            "remove": True,
            "network_mode": "host",
            "log_config": {
                "Type": "k8s-file",
                "Config": {
                    "path": self.cfg.logs_path
                }
            }
        }

        if gpuidx is not None:
            kwargs["devices"] = [f"nvidia.com/gpu={gpuidx}"]

        container = self.pclient.containers.create(**kwargs)
        container.start()
        self.container = container

        self, 
        gpuidx: int | None,
    ) -> None:
        os.makedirs(self.cfg.tags_dir, exist_ok=True)
        volumes = [
            {
                "source": self.cfg.tags_dir,
                # convention for containers to store tags in /elv/tags
                "target": "/elv/tags",
                "type": "bind",
            },
            {
                "source": self.cfg.cache_dir,
                # convention for python modules to store cache in /root/.cache
                "target": "/root/.cache",
                "type": "bind",
                "read_only": False
            }
        ]

        for f in self.cfg.file_args:
            if not os.path.exists(f):
                raise FileNotFoundError(f"File {f} not found")
            elif not os.path.isfile(f):
                raise IsADirectoryError(f"{f} is a directory")
            elif not os.path.isabs(f):
                raise ValueError(f"{f} must be an absolute path")
            # mount the file
            volumes.append({
                "source": f,
                "target": f"/elv/{os.path.basename(f)}",
                "type": "bind",
                "read_only": True
            })

        kwargs = {
            "image": self.cfg.model_config.image,
            "command": [f"{os.path.basename(f)}" for f in self.cfg.file_args] + ["--config", f"{json.dumps(self.cfg.run_config)}"],
            "mounts": volumes,
            "remove": True,
            "network_mode": "host",
            "log_config": {
                "Type": "k8s-file",
                "Config": {
                    "path": self.cfg.logs_path
                }
            }
        }

        if gpuidx is not None:
            kwargs["devices"] = [f"nvidia.com/gpu={gpuidx}"]

        container = self.pclient.containers.create(**kwargs)
        container.start()
        self.container = container

    def stop(self) -> None:
        if not self.container:
            return
        if self.container.status == "running":
            self.container.stop(timeout=5)
        if self.is_running():
            logger.warning("Container did not stop in time, killing it", extra={"container_id": self.container.id, "handle": self.name()})
            self.container.kill()

    def is_running(self) -> bool:
        if self.container is None:
            return False
        self.container.reload()
        return self.container.status == "running" or self.container.status == "created"

    def exit_code(self) -> int | None:
        """Returns exit code if available, else None"""
        if self.container is None:
            return None
        self.container.reload()
        if self.container.status == "exited":
            return self.container.attrs["State"]["ExitCode"]
        return None

    def tags(self) -> list[ModelOutput]:
        """
        Get output tags generated by the running container so far.

        NOTE: this list should be append only and no previous outputs should be modified.
        This responsibility is on the container implementation.
        """
        if not os.path.exists(self.cfg.tags_dir):
            # hasn't started
            return []
        
        tag_files = []
        for fpath in os.listdir(self.cfg.tags_dir):
            tag_files.append(os.path.join(self.cfg.tags_dir, fpath))

        return self._files_to_tags(tag_files)
    
    def name(self) -> str:
        return f"{self.cfg.id}_{self.cfg.model_config.image}"

    def required_resources(self) -> SystemResources:
        return copy(self.cfg.model_config.resources)

    def _files_to_tags(self, tagged_files: list[str]) -> list[ModelOutput]:

        if self.file_type == "image":
            return self._load_image_tags(tagged_files)
        elif self.file_type == "video" or self.file_type == "audio":
            return self._load_video_tags(tagged_files)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def _load_image_tags(self, tagged_files: list[str]) -> list[ModelOutput]:
        outputs = []
        for tagged_file in tagged_files:
            source = self._source_from_tag_file(tagged_file)
            image_tags = []
            try:
                with open(tagged_file, 'r') as f:
                    image_tags = json.load(f)
            except Exception as e:
                logger.error(f"Error loading image tags from {tagged_file}: {e}")
                continue
            outputs.append(self._output_from_image_tags(source, image_tags))
        return outputs

    def _output_from_image_tags(self, source_image: str, image_tags: list[dict]) -> ModelOutput:
        tags = []
        for image_tag_data in image_tags:
            tags.append(Tag(
                start_time=0,
                end_time=0,
                text=image_tag_data.get("text", ""),
                additional_info={
                    "confidence": image_tag_data.get("confidence", 0.0),
                    "box": image_tag_data.get("box", [])
                },
                source="",
                jobid=""
            ))

        return ModelOutput(
            source_media=source_image,
            tags=tags
        )
        
    def _load_video_tags(self, tagged_files: list[str]) -> list[ModelOutput]:
        
        source_to_tagfiles = {}

        for tagged_file in tagged_files:
            source_media = self._source_from_tag_file(tagged_file)
            if source_media not in source_to_tagfiles:
                source_to_tagfiles[source_media] = []
            source_to_tagfiles[source_media].append(tagged_file)

        outputs = []
        for source_media, tag_files in source_to_tagfiles.items():
            model_out = self._output_from_tags(source_media, tag_files)
            if model_out:
                outputs.append(model_out)

        return outputs

    def _output_from_tags(self, source_video: str, tag_files: list[str]) -> ModelOutput | None:
        ftype = get_file_type(source_video)

        if ftype == "video":
            fps = get_fps(source_video)
        else:
            fps = None
    
        vid_tags = None
        frame_tags = None

        for tag_file in tag_files:
            try:
                if tag_file.endswith("_tags.json"):
                    with open(tag_file, 'r') as f:
                        vid_tags = json.load(f)
                elif tag_file.endswith("_frametags.json"):

                    with open(tag_file, 'r') as f:
                        frame_tags = json.load(f)
            except Exception as e:
                logger.error(f"Error loading tags from {tag_file}: {e}")
                continue
        
        if vid_tags is None:
            return None

        # Convert video tags to Tag objects with enhanced additional_info
        tags = []

        for video_tag_data in vid_tags:
            # Create base Tag object from video tag
            tag = Tag(
                start_time=video_tag_data.get("start_time", 0),
                end_time=video_tag_data.get("end_time", 0),
                text=video_tag_data.get("text", ""),
                additional_info={},
                source="",
                jobid=""
            )

            if frame_tags:
                if ftype != "video":
                    raise ValueError("Frame tags can only be associated with video files")
                # Find overlapping frame tags with matching text
                assert fps is not None
                overlapping_frame_tags = self._find_overlapping_frame_tags(
                    tag, frame_tags, fps
                )
                
                # Enhance additional_info with frame tag data
                if overlapping_frame_tags:
                    frame_info = {}
                    for ftag in overlapping_frame_tags:
                        frame_info[ftag.frame_idx] = {
                            "confidence": ftag.confidence,
                            "box": ftag.box
                        }
                    tag.additional_info["frame_tags"] = frame_info

            tags.append(tag)

        return ModelOutput(
            source_media=source_video,
            tags=tags
        )

    def _source_from_tag_file(self, tagfile: str) -> str:
        """
        Extract the source from the tag file name.
        """
        basename = os.path.basename(tagfile)
        # remove _tags, _frametags, or _imagetags suffix
        path_parts = basename.split("_")
        if len(path_parts) < 2:
            raise ValueError(f"Invalid tag file name: {basename}")
        suffix = path_parts[-1]
        if suffix not in ["tags.json", "frametags.json", "imagetags.json"]:
            raise ValueError(f"Invalid tag file suffix: {suffix}")
        original_filebase = "_".join(path_parts[:-1])

        return self.basename_to_source[original_filebase]

    def _find_overlapping_frame_tags(
        self, 
        video_tag: Tag, 
        frame_tags_data: dict, 
        fps: float
    ) -> list[FrameTag]:
        overlapping_tags = []
        for fidx, ftags in frame_tags_data.items():
            frame_time = (int(fidx) / fps) * 1000
            if video_tag.start_time <= frame_time < video_tag.end_time:
                for ftag in ftags:
                    if ftag.get("text", "") == video_tag.text:
                        overlapping_tags.append(FrameTag(
                            frame_idx=fidx,
                            confidence=ftag.get("confidence", 0.0),
                            box=ftag.get("box", None),
                            text=ftag.get("text", "")
                        ))
        return overlapping_tags

    def _find_common_root(self, filepaths: list[str]) -> str:
        """Find the common root directory for all files"""
        if not filepaths:
            raise ValueError("No files provided")
        
        # Get absolute paths
        abs_paths = [os.path.abspath(f) for f in filepaths]
        
        # Find common prefix
        common_prefix = os.path.commonpath(abs_paths)
        
        # If common prefix is a file (only one file), use its directory
        if os.path.isfile(common_prefix):
            common_prefix = os.path.dirname(common_prefix)
        
        return common_prefix

    def _calculate_max_depth(self, filepaths: list[str], root: str) -> int:
        """Calculate maximum depth of files relative to root directory"""
        max_depth = 0
        
        for filepath in filepaths:
            rel_path = os.path.relpath(filepath, root)
            # Count directory separators to determine depth
            depth = rel_path.count(os.sep)
            max_depth = max(max_depth, depth)
        
        return max_depth

class ContainerRegistry:
    """
    Get runnable containers through identifier
    """

    def __init__(self, cfg: RegistryConfig):
        self.pclient = PodmanClient()
        self.cfg = cfg
        os.makedirs(self.cfg.base_dir, exist_ok=True)
        os.makedirs(self.cfg.cache_dir, exist_ok=True)

    def get(self, req: ContainerRequest) -> TagContainer:
        if req.job_id is not None:
            jobid = req.job_id
        else:
            jobid = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + str(uuid.uuid4())[:6]
            logger.warning(f"User request {req} did not give jobid, generating default: {jobid}")

        jobpath = os.path.join(self.cfg.base_dir, req.model_id, jobid)
        tags_path = os.path.join(jobpath, 'tags')
        logs_path = os.path.join(jobpath, 'log.out')

        cache_path = self.cfg.cache_dir

        modelcfg = self.cfg.model_configs.get(req.model_id)
        if not modelcfg:
            raise MissingResourceError(f"Model {req.model_id} not found")

        ccfg = ContainerSpec(
            id=jobid,
            file_args=req.file_args,
            run_config=req.run_config,
            logs_path=logs_path,
            cache_dir=cache_path,
            tags_dir=tags_path,
            model_config=modelcfg
        )

        return TagContainer(self.pclient, ccfg)

    def get_model_config(self, model: str) -> ModelConfig:
        return deepcopy(self.cfg.model_configs[model])

    def services(self) -> list[str]:
        """
        Returns a list of available services
        """
        # TODO: check if the image exists
        return list(self.cfg.model_configs.keys())