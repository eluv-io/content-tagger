from podman import PodmanClient
from src.common.logging import logger
import json
import os
from copy import copy
import socket
from urllib.parse import unquote
import time

from src.common.errors import BadRequestError
from src.tag_containers.model import *
from src.tag_containers.model import ContainerInfo, ContainerOutput, Progress, Error

logger = logger.bind(name="TagContainer")

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

        # check that no file has the same basename
        self.basename_to_source = {os.path.basename(f): f for f in self.media_files}
        if len(self.basename_to_source) != len(self.media_files):
            raise BadRequestError("Files must have unique basenames")
        self.pclient = pclient
        self.container = None
        self.stdin_socket = None
        self.eof = False

    def start(self, gpuidx: int | None) -> None:
        if self.eof:
            raise RuntimeError("Container has already received EOF, cannot start again.")

        output_dir = os.path.dirname(self.cfg.output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = os.path.basename(self.cfg.output_path)

        volumes = [
            {
                "source": output_dir,
                "target": "/elv/output",
                "type": "bind",
            },
            {
                "source": self.cfg.cache_dir,
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

        cmd = self._get_args(output_filename)

        kwargs = {
            "image": self.cfg.model_config.image,
            "command": cmd,
            "mounts": volumes,
            "remove": True,
            "network_mode": "host",
            "pids_limit": -1,
            "stdin_open": True,
            "tty": False,
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

        timeout = 10
        start = time.time()
        has_started = False
        while time.time() - start < timeout:
            if self.is_running():
                has_started = True
                break
            time.sleep(0.1)
        if not has_started:
            raise RuntimeError(f"Container did not start in time: {self.name()}")
        
        self.stdin_socket = self._open_container_stdin(self.container)
        if self.media_files:
            self.add_media(self.media_files)

    def add_media(self, new_media: list[str]) -> None:
        if not self.is_running():
            logger.warning("Container is not running, cannot add media", extra={"handle": self.name()})
            return
        
        if len(new_media) == 0:
            return
        
        for f in new_media:
            self.basename_to_source[os.path.basename(f)] = f

        assert self.stdin_socket is not None
        
        media_files = self._get_relative_paths(new_media)
        logger.info(f"Adding {len(media_files)} media files to container: {media_files[:2]}...")
        msg = "\n".join(media_files) + "\n"
        self.stdin_socket.sendall(msg.encode())

    def send_eof(self) -> None:
        self.eof = True
        if self.stdin_socket:
            try:
                try:
                    self.stdin_socket.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                finally:
                    self.stdin_socket.close()
            except Exception as e:
                logger.opt(exception=e).error("Error closing stdin socket", extra={"handle": self.name()})
        else:
            logger.warning("No stdin socket to close", extra={"handle": self.name()})

    def stop(self) -> None:
        self.send_eof()
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
        return self.container.status == "running"

    def exit_code(self) -> int | None:
        """Returns exit code if available, else None"""
        if self.container is None:
            return None
        self.container.reload()
        if self.container.status == "exited":
            return self.container.attrs["State"]["ExitCode"]
        return None

    def tags(self) -> list[ModelTag]:
        """
        Get output tags generated by the running container so far.
        Reads the JSONL output file and returns all "tag" type messages as ModelTag objects.
        """
        return self._parse_output().tags

    def errors(self) -> list[Error]:
        """Get error messages reported by the container."""
        return self._parse_output().errors

    def progress(self) -> list[Progress]:
        """Get progress reports from the container (which files are fully processed)."""
        return self._parse_output().progress
    
    def name(self) -> str:
        """A human friendly name for the container, useful for logging"""
        return f"{self.cfg.id}_{self.cfg.model_config.image}"

    def required_resources(self) -> SystemResources:
        """Returns the system resources required by this container to run."""
        return copy(self.cfg.model_config.resources)

    def info(self) -> ContainerInfo:
        """Returns image annotations and the running container ID (if started)."""
        image = self.pclient.images.get(self.cfg.model_config.image)
        annotations = image.attrs.get("Annotations", {})
        return ContainerInfo(image_name=self.cfg.model_config.image, annotations=annotations)

    def _parse_output(self) -> ContainerOutput:
        """Parse the JSONL output file."""
        tags = []
        progress = []
        errors = []

        if not os.path.exists(self.cfg.output_path):
            return ContainerOutput(tags=tags, progress=progress, errors=errors)

        with open(self.cfg.output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type == "tag":
                    source_basename = data.get("source_media")
                    if not source_basename:
                        raise ValueError("Missing source_media in container output tag")
                    full_source = self.basename_to_source[source_basename]
                    tags.append(ModelTag(
                        start_time=round(data.get("start_time", 0) * 1000),
                        end_time=round(data.get("end_time", 0) * 1000),
                        text=data.get("tag", ""),
                        source_media=full_source,
                        model_track=data.get("track", ""),
                        frame_info=data.get("frame_info"),
                        additional_info=data.get("additional_info"),
                    ))
                elif msg_type == "progress":
                    progress.append(Progress(source_media=data.get("source_media", "")))
                elif msg_type == "error":
                    errors.append(Error(message=data.get("message", ""), source_media=data.get("source_media")))

        return ContainerOutput(tags=tags, progress=progress, errors=errors)
    
    def _get_relative_paths(self, media_files: list[str]) -> list[str]:
        # Calculate paths from perspective of the container working directory "/elv"
        relative_paths = []
        for f in media_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"File {f} not found")
            elif not os.path.isfile(f):
                raise IsADirectoryError(f"{f} is a directory")
            elif not os.path.isabs(f):
                raise ValueError(f"{f} must be an absolute path")
            elif not f.startswith(self.media_dir):
                raise ValueError(f"{f} is not in media directory {self.media_dir}")
            
            rel_path = os.path.relpath(f, self.media_dir)
            relative_paths.append(f"media/{rel_path}")
        return relative_paths
    
    def _get_args(self, output_filename: str) -> list[str]:
        """Get command line arguments for the container"""
        return [
            "--output-path", f"/elv/output/{output_filename}",
            "--params", json.dumps(self.cfg.run_config),
        ]

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

    def _open_container_stdin(self, container):    
        ## assume podman is on the same machine via unix socket
        consocket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        consocket.connect(unquote(container.client.base_url.netloc))

        ## we have to do this manually, because once the podman socket server accepts the POST
        ## it then converts the socket into a "raw" socket for writing directly to the container's stdin
        msg = f"POST /v5.4.0/libpod/containers/{container.id}/attach?stdin=1&stdout=0&stderr=0 HTTP/1.0\r\n\r\n".encode()
        consocket.sendall(msg)

        ## response looks like this: 'HTTP/1.1 200 OK\r\nContent-Type: application/vnd.docker.raw-stream\r\n\r\n'
        response = consocket.recv(4096)
        
        logger.debug(f"socket response: {response}")
        
        ## make sure we successfully opened the connection
        ## be slightly flexible but otherwise pretty strict
        if response[0:15] != bytes("HTTP/1.1 200 OK", "utf-8") and response[0:15] != bytes("HTTP/1.0 200 OK", "utf-8"):
            raise Exception(f"Did not successfully open stdin for container: {response}")
        
        return consocket
