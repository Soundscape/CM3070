"""Module containing cache implementations"""

import json

from pymemcache.client import base


class MemoryCache():
    """A cache implementation using Memcached
    """

    def __init__(self) -> None:
        self.cache = base.Client(('localhost', 11211))
        self.cache.flush_all()

    def set(self, key: str, value: any, expires: int = 3600):
        """Set a key to a value for a number of seconds

        Args:
            key (str): The key
            value (any): The value
            expires (int, optional): The number of seconds before the entry expires. Defaults to 3600.
        """
        if isinstance(value, list):
            json_val = str.join(',', [d.json() for d in value])
            json_val = f'[{json_val}]'
        else:
            json_val = value.json()
        self.cache.set(key, json_val, expires)

    def get(self, key: str, converter):
        """Retrieve a value from the cache using its key

        Args:
            key (str): The key
            converter (function): A converter function to transform the cached value into a type

        Returns:
            Object: The retrieved value
        """
        json_val = self.cache.get(key)
        if json_val is None:
            return None
        value = json.loads(json_val)

        if isinstance(value, list):
            return [converter(d) for d in value]

        return converter(value)
