import concurrent.futures
import hashlib
import json
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, List, Tuple
from urllib.parse import urlparse

import s3fs
from tqdm import tqdm

from cookbook.aliases import SourceConfig
from olmo_core.aliases import PathOrStr
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import get_file_size, is_url, normalize_path
from olmo_core.utils import OLMoEnvironmentError

logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.WARNING)


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

    token_counts = defaultdict(int)

    client_kwargs = {}
    for source in source_configs:
        for path in source.paths:
            parsed = urlparse(path)
            if parsed.scheme == "s3":
                continue
            elif parsed.scheme == "weka":
                client_kwargs["endpoint_url"] = os.environ.get("WEKA_ENDPOINT_URL")
            elif parsed.scheme == "gs":
                client_kwargs["endpoint_url"] = "https://storage.googleapis.com"
                client_kwargs["key"] = os.environ.get("GS_INTEROP_KEY")
                client_kwargs["secret"] = os.environ.get("GS_INTEROP_SECRET")
    fs = s3fs.S3FileSystem(client_kwargs={**client_kwargs})

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        for source in source_configs:
            globs = [path for path in source.paths if "*" in path]
            paths = [path for path in source.paths if path not in globs]
            source.paths = paths + expand_globs(fs, globs)

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


def expand_globs(s3: s3fs.S3FileSystem, sources: List[str]) -> Any:
    results = []

    for source in sources:
        if is_url(source):
            logger.info(f"Expanding remote glob '{source}'...")
            results.extend(_expand_remote(source, s3))
        else:
            logger.info(f"Expanding local glob '{source}'...")
            results.extend(_expand_local(source))

    return results


def _expand_local(pattern: str) -> List[str]:
    """
    Expand a local glob pattern.
    """
    from glob import glob

    logger.info(f"Expanding '{pattern}'...")
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)

    return [normalize_path(match) for match in matches]


def _expand_remote(pattern: str, fs: s3fs.S3FileSystem) -> List[str]:
    """
    Expand a remote glob pattern.
    """
    parsed = urlparse(pattern)

    if parsed.scheme == "s3":
        return [f"s3://{obj}" for obj in fs.glob(pattern)]
    elif parsed.scheme == "r2":
        raise NotImplementedError("'r2' types are not currently supported")
    elif parsed.scheme == "weka":
        return [f"weka://{obj}" for obj in fs.glob(pattern)]
    elif parsed.scheme == "gs":
        raise NotImplementedError("'gs' types are not currently supported")
    elif parsed.scheme in ("http", "https"):
        raise NotImplementedError("'http' types are not currently supported")
    elif parsed.scheme == "file":
        raise NotImplementedError("'file' types are not currently supported")
    else:
        raise NotImplementedError(f"Glob expansion is not currently supported for '{parsed.scheme}' files")


def normalize_source_paths(sources: List[SourceConfig]) -> List[SourceConfig]:
    """
    Normalize the paths in a SourceConfig object.
    """
    normalized = []

    for source in sources:
        source_paths = []
        for path in source.paths:
            if is_url(path):
                parsed = urlparse(path)
                if parsed.scheme == "s3":
                    source_paths.append(path)
                elif parsed.scheme == "weka":
                    source_paths.append(normalize_path(path.replace("weka://", "/weka/")))
                elif parsed.scheme == "gs":
                    source_paths.append(path)
                elif parsed.scheme == "r2":
                    raise NotImplementedError("'r2' types are not currently supported")
                elif parsed.scheme in ("http", "https"):
                    raise NotImplementedError("'http' types are not currently supported")
                else:
                    raise OLMoEnvironmentError(f"Unsupported URL scheme: {parsed.scheme}")
            else:
                source_paths.append(normalize_path(path))

        normalized.append(
            SourceConfig(
                name=source.name,
                paths=source_paths,
                target_ratio=source.target_ratio,
                repetition_factor=source.repetition_factor,
                max_source_ratio=source.max_source_ratio,
            )
        )

    return normalized
