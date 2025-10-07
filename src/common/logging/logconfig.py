import sys
from loguru import logger

logger.remove()
fmt_console = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> | {extra}"
fmt_file    = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message} | {extra}"
logger.add("tagger.log", rotation="10 MB", format=fmt_file)
logger.add(sys.stderr, format=fmt_console, colorize=True)