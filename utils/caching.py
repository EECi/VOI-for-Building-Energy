"""Implementation of cache wrapper with load/save functionality.

Taken from - https://stackoverflow.com/questions/15585493/store-the-cache-to-a-file-functools-lru-cache-in-python-3-2.
"""

from functools import wraps

def cached(func):
    func.cache = {}
    @wraps(func)
    def wrapper(*args):
        key = '-'.join([str(arg) for arg in args])
        try:
            return func.cache[key]
        except KeyError:
            func.cache[key] = result = func(*args)
            return result
    return wrapper