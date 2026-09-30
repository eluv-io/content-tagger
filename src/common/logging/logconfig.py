import re
import sys
from dataclasses import dataclass
from loguru import logger

@dataclass
class LoggingConfig:
    level: str = "INFO"

fmt_console = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> | {extra}"
fmt_file    = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message} | {extra}"

# matches e.g. ?authorization=..., token='...', 'ELV_TOKEN': '...', Authorization: Bearer ...
_SECRET_PATTERN = re.compile(
    r"""(?i)((?<!write_)(?:authorization|token)\\?['"]?\s*[:=]\s*\\?['"]?(?:bearer[+ ])?)[^\s'"\\&,)}\]]+"""
)

def redact(text: str) -> str:
    return _SECRET_PATTERN.sub(r"\1<redacted>", text)

class _RedactingStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, message: str) -> None:
        self._stream.write(redact(message))

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

def configure_logging(cfg: LoggingConfig) -> None:
    logger.remove()
    # diagnose=False: tracebacks would otherwise dump local variable values (including tokens)
    logger.add(_RedactingStream(sys.stderr), format=fmt_console, level=cfg.level.upper(), diagnose=False)

configure_logging(LoggingConfig())
