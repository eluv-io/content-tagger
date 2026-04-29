import threading
from contextlib import contextmanager

StreamKey = tuple[str, str]

class FetchRateLimiter:
    def __init__(self, max_concurrent: int):
        self._sem = threading.Semaphore(max_concurrent)
        self._locks: dict[StreamKey, threading.Lock] = {}
        self._locks_mu = threading.Lock()

    def _get_lock(self, key: StreamKey) -> threading.Lock:
        with self._locks_mu:
            return self._locks.setdefault(key, threading.Lock())

    @contextmanager
    def permit(self, key: StreamKey):
        # throttle total API calls
        self._sem.acquire()
        try:
            # avoid downloading the same stream from two places
            lock = self._get_lock(key)
            with lock:
                yield
        finally:
            self._sem.release()