import os
from pathlib import Path

from platformdirs import user_cache_dir


def get_cache_path(dashboard) -> Path:
    cache_dir = user_cache_dir("cookbook")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    path = Path(cache_dir) / dashboard
    return path
