import concurrent.futures
import logging
import os
import pathlib
from collections import defaultdict
from typing import Tuple
from typing import Any, List
from urllib.parse import urlparse

import s3fs
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import is_url

import s3fs
from olmo_core.aliases import PathOrStr
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import get_file_size
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.WARNING)


import hashlib
import json

from cookbook.aliases import SourceConfig


def _bytes_to_tokens(num_bytes: int, dtype: NumpyDatasetDType) -> int:
    """
    Convert bytes to tokens based on the dtype.
    """
    npdtype = dtype.as_np_dtype()
    return num_bytes // npdtype(int(0)).itemsize


def _count_tokens_for_file(path: PathOrStr, dtype: NumpyDatasetDType) -> int:
    return _bytes_to_tokens(get_file_size(path), dtype)


def get_token_counts_and_ratios(
    source_configs: list[SourceConfig], dtype: NumpyDatasetDType, use_cache: bool
) -> Tuple[dict[str, float], int]:
    config_hash = hashlib.md5(
        json.dumps(
            [(sc.name, sc.paths) for sc in source_configs],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    cache_path = pathlib.Path(f"/tmp/olmo-cookbook/priors_cache_{config_hash}.json")
    if use_cache:
        try:
            with open(cache_path, "r") as f:
                logger.info(
                    "Source distribution cache found, using cached values! This can be disabled by setting use_cache=False."
                )
                obj = json.load(f)
                return (obj["relative_sizes"], obj["total_tokens"])
        except FileNotFoundError:
            logger.info("No cache file found, calculating from source files...")

    fs = s3fs.S3FileSystem(anon=False)
    token_counts = defaultdict(int)

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        for source in source_configs:
            source.paths = expand_globs(fs, source.paths)

        futures = {
            executor.submit(_count_tokens_for_file, path, dtype): source
            for source in source_configs
            for path in source.paths
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            source_future = futures[future]
            try:
                result = future.result()
                token_counts[source_future.name] += result
            except Exception as e:
                logger.info(f"Error processing {source_future.name}: {str(e)}")
                token_counts[source_future.name] = 0

    # Calculate relative sizes
    total_tokens = sum(token_counts.values())

    if total_tokens == 0:
        raise Exception(f"Error processing config, no tokens found!")

    relative_sizes = {path: count / total_tokens for path, count in token_counts.items()}

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"relative_sizes": relative_sizes, "total_tokens": total_tokens}, f)

    return (relative_sizes, total_tokens)


def expand_globs(s3: s3fs.S3FileSystem, paths: List[str]) -> Any:
    results = []

    for path in paths:
        if is_url(path):
            parsed = urlparse(str(path))
            if parsed.scheme in ("s3", "r2", "weka"):
                results.extend([f"s3://{obj}" for obj in s3.glob(path)])
            elif parsed.scheme == "gs":
                raise NotImplementedError("'gs' types are not currently supported")
            elif parsed.scheme in ("http", "https"):
                raise NotImplementedError("'http' types are not currently supported")
            elif parsed.scheme == "file":
                raise NotImplementedError("'file' types are not currently supported")
            else:
                raise NotImplementedError(
                    f"Glob expansion is not currently supported for '{parsed.scheme}' files"
                )
        else:
            raise NotImplementedError("Glob expansion is only supported for URLs")

    return results
