from loguru import logger

logger.add("tagger.log", rotation="10 MB", format="{time} | {level} | {message} | {extra}")