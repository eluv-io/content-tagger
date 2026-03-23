from podman import PodmanClient
from src.common.logging import logger
import json
import os
from copy import copy
import socket
from urllib.parse import unquote
import threading
import time

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

        self.media_dir = cfg.media_dir

        self.pclient = pclient
        self.container = None
        self.stdin_socket = None
        self.eof = False

        # used for converting source_media field in model outputs back to the path on the host filesystem
        self.basename_to_source = {}

        self._media_buffer: list[str] = []
        self._lock = threading.Lock()

    def start(self, gpuidx: int | None) -> None:
        with self._lock:
            self._start(gpuidx)

    def add_media(self, new_media: list[str]) -> None:
        with self._lock:
            self._add_media(new_media)

    def send_eof(self) -> None:
        with self._lock:
            self._send_eof()

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def is_running(self) -> bool:
        with self._lock:
            return self._is_running()

    def exit_code(self) -> int | None:
        with self._lock:
            return self._exit_code()

    def tags(self) -> list[ModelTag]:
        with self._lock:
            return self._tags()

    def errors(self) -> list[Error]:
        with self._lock:
            return self._errors()

    def progress(self) -> list[Progress]:
        with self._lock:
            return self._progress()

    def name(self) -> str:
        with self._lock:
            return self._name()

    def required_resources(self) -> SystemResources:
        with self._lock:
            return self._required_resources()

    def info(self) -> ContainerInfo:
        with self._lock:
            return self._info()

    def _start(self, gpuidx: int | None) -> None:
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
            if self._is_running():
                has_started = True
                break
            time.sleep(0.1)
        if not has_started:
            raise RuntimeError(f"Container did not start in time: {self._name()}")
        
        self.stdin_socket = self._open_container_stdin(self.container)

        self._flush_media()
        if self.eof:
            # then we've already queued an eof and we need to 
            self._send_eof()

    def _add_media(self, new_media: list[str]) -> None:
        """Buffer media files to be sent to the container on the next flush_media call."""
        if len(new_media) == 0:
            return
        for fpath in new_media:
            assert os.path.dirname(fpath) == self.media_dir

        for f in new_media:
            self.basename_to_source[os.path.basename(f)] = f
        self._media_buffer.extend(new_media)

        if self._is_running():
            self._flush_media()

    def _send_eof(self) -> None:
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
                logger.opt(exception=e).error("Error closing stdin socket", handle=self._name())
        else:
            logger.warning("No stdin socket to close", handle=self._name())

    def _stop(self) -> None:
        self._send_eof()
        if not self.container:
            return
        if self.container.status == "running":
            self.container.stop(timeout=5)
        if self._is_running():
            logger.warning("Container did not stop in time, killing it", extra={"container_id": self.container.id, "handle": self._name()})
            self.container.kill()

    def _is_running(self) -> bool:
        if self.container is None:
            return False
        self.container.reload()
        return self.container.status == "running"

    def _exit_code(self) -> int | None:
        """Returns exit code if available, else None"""
        if self.container is None:
            return None
        self.container.reload()
        if self.container.status == "exited":
            return self.container.attrs["State"]["ExitCode"]
        return None

    def _tags(self) -> list[ModelTag]:
        """
        Get output tags generated by the running container so far.
        Reads the JSONL output file and returns all "tag" type messages as ModelTag objects.
        """
        return self._parse_output().tags

    def _errors(self) -> list[Error]:
        """Get error messages reported by the container."""
        return self._parse_output().errors

    def _progress(self) -> list[Progress]:
        """Get progress reports from the container (which files are fully processed)."""
        return self._parse_output().progress

    def _name(self) -> str:
        """A human friendly name for the container, useful for logging"""
        return f"{self.cfg.id}_{self.cfg.model_config.image}"

    def _required_resources(self) -> SystemResources:
        """Returns the system resources required by this container to run."""
        return copy(self.cfg.model_config.resources)

    def _info(self) -> ContainerInfo:
        """Returns image annotations and the running container ID (if started)."""
        image = self.pclient.images.get(self.cfg.model_config.image)
        annotations = image.attrs.get("Annotations", {})
        return ContainerInfo(image_name=self.cfg.model_config.image, annotations=annotations)

    def _flush_media(self) -> None:
        """Send all buffered media files to the container's stdin."""
        buffered = self._media_buffer
        self._media_buffer = []

        assert self.stdin_socket is not None

        media_files = self._get_relative_paths(buffered)
        logger.info(f"Flushing {len(media_files)} media files to container: {media_files[:2]}...")
        msg = "\n".join(media_files) + "\n"
        self.stdin_socket.sendall(msg.encode())

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

                msg_type = msg["type"]
                data = msg["data"]

                # get the source name if it exists
                local_media_path = None
                source_media = data.get("source_media")

                if not source_media and msg_type != "error":
                    raise ValueError(f"Missing source_media in container output message: {msg}")
                elif msg_type == "error" and not source_media:
                    # allow source_media to be None
                    pass
                else:
                    source_basename = os.path.basename(source_media)
                    local_media_path = self.basename_to_source[source_basename]

                if msg_type == "tag":
                    # placate type checker
                    assert local_media_path is not None                   
                    tags.append(ModelTag(
                        start_time=data.get("start_time", 0),
                        end_time=data.get("end_time", 0),
                        text=data.get("tag", ""),
                        source_media=local_media_path,
                        model_track=data.get("track", ""),
                        frame_info=data.get("frame_info"),
                        additional_info=data.get("additional_info"),
                    ))
                elif msg_type == "progress":
                    # placate type checker
                    assert local_media_path is not None  
                    progress.append(Progress(source_media=local_media_path))
                elif msg_type == "error":
                    errors.append(Error(message=data.get("message", ""), source_media=local_media_path))
                else:
                    raise ValueError(f"Got unexpected message type: {msg}")

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
