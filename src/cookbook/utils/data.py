import concurrent.futures
import hashlib
import json
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union
from urllib.parse import urlparse

import gcsfs
import s3fs
from olmo_core.aliases import PathOrStr
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import get_file_size, is_url, normalize_path
from olmo_core.utils import OLMoEnvironmentError
from tqdm import tqdm

from cookbook.aliases import SourceConfig

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


def get_leaf_configs(source_config: SourceConfig) -> List[Tuple[str, List[str]]]:
    """Return a list of (name, paths) tuples representing the leaf nodes.
       This is important when we have data sources that are divided into topics.
    """
    if source_config.topics:
        return [
            (f"{source_config.name}:{topic.name}", topic.paths)
            for topic in source_config.topics
        ]
    else:
        return [(source_config.name, source_config.paths)]

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

    filesystems = {}
    leaf_configs: list[SourceConfig] = [] 
    for sc in source_configs:                         
        leaf_configs.extend(
            SourceConfig(name=leaf_name, paths=leaf_paths)
            for leaf_name, leaf_paths in get_leaf_configs(sc)
        )
    source_configs = leaf_configs

    # Pre-check each source for mixed schemes and create appropriate filesystem clients
    for source in source_configs:
        schemes = {urlparse(path).scheme for path in source.paths}

        # Check for mixed schemes within a source
        if len(schemes) > 1 and any(scheme for scheme in schemes):
            raise OLMoEnvironmentError(
                f"Mixed URL schemes in source '{source.name}': {schemes}. Each source must use a consistent scheme."
            )

        # Get the scheme (or None for local paths)
        scheme = next(iter(schemes)) if schemes and next(iter(schemes)) else "local"

        if scheme not in filesystems:
            filesystems[scheme] = get_filesystem_for_scheme(scheme)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for source in source_configs:
            # Get the appropriate filesystem for this source
            scheme = next(iter({urlparse(path).scheme for path in source.paths}), "local")
            fs = filesystems.get(scheme)

            globs = [path for path in source.paths if "*" in path]
            paths = [path for path in source.paths if path not in globs]
            source.paths = paths + expand_globs(fs, globs) if globs else paths

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


def expand_globs(
    fs: Optional[Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]] = s3fs.S3FileSystem(), sources: List[str] = []
) -> Any:
    results = []

    for source in sources:
        if is_url(source):
            results.extend(_expand_remote(source, fs))
        else:
            results.extend(_expand_local(source))

    # Filter the globs from the expanded list
    return [r for r in results if "*" not in r]


def _expand_local(pattern: str) -> List[str]:
    """
    Expand a local glob pattern.
    """
    from glob import glob

    logger.info(f"Expanding '{pattern}'...")
    matches = sorted(glob(pattern, recursive=True))

    if not matches:
        raise FileNotFoundError(pattern)

    return [normalize_path(match) for match in matches]


def _expand_remote(pattern: str, fs: Optional[Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]]) -> List[str]:
    """
    Expand a remote glob pattern.
    """
    if not fs:
        fs = s3fs.S3FileSystem()

    parsed = urlparse(pattern)
    logger.info(f"Expanding remote glob '{pattern}'...")

    if parsed.scheme == "s3":
        return [f"s3://{obj}" for obj in fs.glob(pattern)]
    elif parsed.scheme == "weka":
        return [f"weka://{obj}" for obj in fs.glob(pattern.replace("weka://", "s3://"))]
    elif parsed.scheme == "gs":
        return [f"gs://{obj}" for obj in fs.glob(pattern)]
    elif parsed.scheme == "r2":
        raise NotImplementedError("'r2' types are not currently supported")
    elif parsed.scheme in ("http", "https"):
        raise NotImplementedError("'http' types are not currently supported")
    elif parsed.scheme == "file":
        raise NotImplementedError("Remote 'file' types are not currently supported")
    else:
        raise NotImplementedError(f"Glob expansion is not currently supported for '{parsed.scheme}' files")


def normalize_source_paths(sources: List[SourceConfig], expand: bool = False) -> List[SourceConfig]:
    """
    Normalize the paths in a SourceConfig object.
    """
    normalized = []

    for source in sources:
        source_paths = []
        schemes = set()

        for path in source.paths:
            if is_url(path):
                parsed = urlparse(path)
                schemes.add(parsed.scheme)
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
                schemes.add("local")

        # Get filesystem if we're expanding globs and paths exist
        fs = None
        if expand and source_paths:
            scheme = next(iter(schemes)) if schemes else "local"
            fs = get_filesystem_for_scheme(scheme)

        normalized.append(
            SourceConfig(
                name=source.name,
                paths=expand_globs(fs=fs, sources=source_paths) if expand else source_paths,
                target_ratio=source.target_ratio,
                repetition_factor=source.repetition_factor,
                max_source_ratio=source.max_source_ratio,
            )
        )

    return normalized


def get_filesystem_for_scheme(scheme: str):
    """
    Get the appropriate filesystem for a given URL scheme.

    Args:
        scheme: The URL scheme (e.g., 's3', 'gs', 'local', 'weka')

    Returns:
        The appropriate filesystem object for the scheme or None for local paths

    Raises:
        OLMoEnvironmentError: If the scheme is not supported or not configured correctly
        NotImplementedError: If the scheme is recognized but not currently supported
    """
    if scheme in ("s3", "weka"):
        client_kwargs = {}
        profile_name = os.environ.get("AWS_PROFILE", None)

        if scheme == "weka":
            profile_name = "WEKA"
            client_kwargs["endpoint_url"] = os.environ.get("WEKA_ENDPOINT_URL")

        return s3fs.S3FileSystem(client_kwargs={**client_kwargs}, profile=profile_name)

    elif scheme == "gs":
        try:
            gs_project = os.environ.get("GOOGLE_CLOUD_PROJECT", None)

            if not gs_project:
                raise OLMoEnvironmentError("GOOGLE_CLOUD_PROJECT environment variable is not set!")

            try:
                return gcsfs.GCSFileSystem(token="google_default")
            except Exception as e:
                logger.warning(
                    f"Failed to create GCS filesystem with default credentials: {str(e)}. Retrying with metadata server..."
                )
                return gcsfs.GCSFileSystem()

        except Exception as e:
            raise OLMoEnvironmentError(
                f"Failed to create GCS filesystem: {str(e)}. Ensure GOOGLE_APPLICATION_CREDENTIALS_JSON and GOOGLE_CLOUD_PROJECT are set correctly."
            )

    elif scheme in ("r2", "http", "https"):
        raise NotImplementedError(f"'{scheme}' scheme is not currently supported")

    elif scheme == "local":
        return None  # No remote filesystem needed for local paths

    else:
        raise OLMoEnvironmentError(f"Unsupported URL scheme: {scheme}")
