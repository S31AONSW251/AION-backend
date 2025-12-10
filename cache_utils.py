"""
Caching utilities for AION.
Provides a `cache_response` decorator that uses Redis when available, falling back to an in-memory dict.
"""
import functools
import time
import json
from typing import Callable

try:
    import redis
except Exception:
    redis = None

_local_cache = {}

def _get_redis_client():
    # Import server lazily to avoid circular imports when server imports this module
    try:
        from . import server as server_module
        return getattr(server_module, 'redis_client', None)
    except Exception:
        return None

def cache_response(ttl: int = 60):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"cache:{func.__name__}:{json.dumps({'args':args,'kwargs':kwargs}, default=str)}"
            rc = _get_redis_client()
            if rc:
                try:
                    v = rc.get(key)
                    if v:
                        return json.loads(v)
                except Exception:
                    pass
            else:
                if key in _local_cache:
                    entry = _local_cache[key]
                    if time.time() - entry['ts'] < ttl:
                        return entry['value']
            # Call original
            result = func(*args, **kwargs)
            try:
                sv = json.dumps(result)
            except Exception:
                try:
                    sv = json.dumps(str(result))
                except Exception:
                    sv = 'null'
            if rc:
                try:
                    rc.set(key, sv, ex=ttl)
                except Exception:
                    pass
            else:
                _local_cache[key] = {'ts': time.time(), 'value': result}
            return result
        return wrapper
    return decorator
