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
_KEYED_SECRET = re.compile(
    r"""(?i)((?<!write_)(?:authorization|token)\\?['"]?\s*[:=]\s*\\?['"]?(?:bearer[+ ])?)[^\s'"\\&,)}\]]+"""
)
# bare fabric tokens (e.g. ascsj_..., atxsjc...) and base64 JSON tokens
_BARE_SECRET = re.compile(r"\b(?:a[a-z]{3}j[a-z]?_?[1-9A-HJ-NP-Za-km-z]{30,}|eyJ[A-Za-z0-9_\-+/=]{30,})")

def redact(text: str) -> str:
    text = _KEYED_SECRET.sub(r"\1<redacted>", text)
    return _BARE_SECRET.sub("<redacted>", text)

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
    logger.add(_RedactingStream(sys.stderr), format=fmt_console, level=cfg.level.upper(), backtrace=True, diagnose=True)

configure_logging(LoggingConfig())
