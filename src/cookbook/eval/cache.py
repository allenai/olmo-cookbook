import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Generic, TypeVar

import smart_open
from platformdirs import user_cache_dir

T = TypeVar("T")
V = TypeVar("V")


@dataclass(frozen=True)
class DatalakeCacheResult(Generic[T]):
    success: bool
    value: T | None


# Singleton instance storage
_DATALAKE_CACHE_INSTANCE = None


@dataclass
class DatalakeCache(Generic[T]):
    cache_dir: str
    invalidate: bool
    do_not_cache: bool

    def __init__(self, invalidate: bool = False, do_not_cache: bool = False):
        self.invalidate = (
            invalidate
            if invalidate is not False
            else (os.environ.get("DATALAKE_CACHE_INVALIDATE", "false").lower() == "true")
        )

        self.do_not_cache = (
            do_not_cache
            if do_not_cache is not False
            else (os.environ.get("DATALAKE_DO_NOT_CACHE", "false").lower() == "true")
        )

        # Set cache_dir
        self.cache_dir = user_cache_dir("datalake", "olmo-cookbook")

        if self.invalidate and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)

            # Check if path exists but is a file instead of a directory
            if os.path.exists(self.cache_dir) and not os.path.isdir(self.cache_dir):
                try:
                    os.remove(self.cache_dir)
                except FileNotFoundError:
                    pass

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _make_cache_path(self, **kwargs) -> str:
        cache_key = hashlib.sha256(json.dumps(kwargs).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.json.gz")

    def get(self, **kwargs) -> DatalakeCacheResult[T]:
        if self.do_not_cache:
            return DatalakeCacheResult(success=False, value=None)

        if os.path.exists(cache_file := self._make_cache_path(**kwargs)) and not self.invalidate:
            with smart_open.open(cache_file, "rt", encoding="utf-8") as f:
                try:
                    return DatalakeCacheResult(success=True, value=json.load(f))
                except (EOFError, json.JSONDecodeError):
                    return DatalakeCacheResult(success=False, value=None)

        return DatalakeCacheResult(success=False, value=None)

    def set(self, value: T, **kwargs) -> DatalakeCacheResult[T]:
        if self.do_not_cache:
            return DatalakeCacheResult(success=False, value=None)

        if not os.path.exists(cache_file := self._make_cache_path(**kwargs)) or self.invalidate:
            with smart_open.open(cache_file, "wt", encoding="utf-8") as f:
                json.dump(value, f)

        return DatalakeCacheResult(success=True, value=value)

    def delete(self, **kwargs) -> None:
        if os.path.exists(cache_file := self._make_cache_path(**kwargs)):
            os.remove(cache_file)


def get_datalake_cache(invalidate: bool = False, do_not_cache: bool = False) -> DatalakeCache:
    """Get or create a singleton instance of DatalakeCache."""
    global _DATALAKE_CACHE_INSTANCE

    if _DATALAKE_CACHE_INSTANCE is None:
        kwargs = {}
        if invalidate is not None:
            kwargs["invalidate"] = invalidate
        if do_not_cache is not None:
            kwargs["do_not_cache"] = do_not_cache
        _DATALAKE_CACHE_INSTANCE = DatalakeCache(**kwargs)

    return _DATALAKE_CACHE_INSTANCE
