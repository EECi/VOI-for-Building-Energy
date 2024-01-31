"""Implementation of cache wrapper with load/save functionality.

Taken from - https://stackoverflow.com/questions/15585493/store-the-cache-to-a-file-functools-lru-cache-in-python-3-2.
"""

from collections.abc import Iterable
from functools import wraps

def cached(func):
    func.cache = {}
    @wraps(func)
    def wrapper(*args):
        # unwrap list of args into key, unwrapping any iterable args
        key = '-'.join([str(v) for arg in args for v in (arg if isinstance(arg,Iterable) else [arg])])
        try:
            return func.cache[key]
        except KeyError:
            func.cache[key] = result = func(*args)
            return result
    return wrapper