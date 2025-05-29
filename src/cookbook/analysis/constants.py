import os
from pathlib import Path

from platformdirs import user_cache_dir


def get_cache_path(dashboard) -> Path:
    cache_dir = user_cache_dir("cookbook")
    path = Path(cache_dir) / dashboard
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
