from dataclasses import dataclass, field as dataclass_field
from typing import TypeVar, Generic
from platformdirs import user_cache_dir
import smart_open
import os
import json
import hashlib
import shutil


T = TypeVar("T")
V = TypeVar("V")



@dataclass(frozen=True)
class DatalakeCacheResult(Generic[T]):
    success: bool
    value: T | None

@dataclass
class DatalakeCache(Generic[T]):
    cache_dir: str = dataclass_field(default_factory=lambda: user_cache_dir("datalake", "olmo-cookbook"))
    invalidate: bool = dataclass_field(default_factory=lambda: os.environ.get("DATALAKE_CACHE_INVALIDATE", "false").lower() == "true")
    do_not_cache: bool = dataclass_field(default_factory=lambda: os.environ.get("DATALAKE_DO_NOT_CACHE", "false").lower() == "true")

    def __post_init__(self):
        if self.invalidate:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _make_cache_path(self, **kwargs) -> str:
        cache_key = hashlib.sha256(json.dumps(kwargs).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.json.gz")

    def get(self, **kwargs) -> DatalakeCacheResult[T]:
        if self.do_not_cache:
            return DatalakeCacheResult(success=False, value=None)

        if os.path.exists(cache_file := self._make_cache_path(**kwargs)) and not self.invalidate:
            with smart_open.open(cache_file, "rt", encoding="utf-8") as f:
                return DatalakeCacheResult(success=True, value=json.load(f))

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
