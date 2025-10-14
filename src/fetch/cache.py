from functools import lru_cache, wraps
from src.common.content import Content

def cache_by_qhash(func):
    """Decorator that caches based on q.qhash and any additional args/kwargs
    
    Especially important because we will likely tag many models on the same content object and this helps to avoid
    unnecessary repeated fabric API calls. 
    """
    @lru_cache(maxsize=128)
    def cached_func(qhash, *args, **kwargs):
        return func(*args, **kwargs)
    
    @wraps(func)
    def wrapper(self, q: Content, *args, **kwargs):
        return cached_func(q.qhash, self, q, *args, **kwargs)
    
    return wrapper