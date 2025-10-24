from loguru import logger
import time

class timeit:
    def __init__(
        self, 
        message: str,
        min_duration: float = 0.0
    ):
        self.message = message
        self.min_duration = min_duration

    def __enter__(self):
        logger.info(f'{self.message}')
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.interval >= self.min_duration:
            logger.debug(f'Finished {self.message}... Elapsed time: {self.interval:.4f} seconds')