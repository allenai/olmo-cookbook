import os
from dataclasses import dataclass, field
from typing import List, Union
from urllib.parse import urlparse

import gcsfs
import s3fs
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.data.types import NumpyDatasetDType

from cookbook.aliases import SourceInstance
from cookbook.utils.data import expand_globs


def _make_weka_fs() -> s3fs.S3FileSystem:

    print("WEKA_ENDPOINT_URL set:", "WEKA_ENDPOINT_URL" in os.environ)
    print("WEKA_ACCESS_KEY_ID set:", "WEKA_ACCESS_KEY_ID" in os.environ)
    print("AWS_ACCESS_KEY_ID set:", "AWS_ACCESS_KEY_ID" in os.environ)
    print("HOME:", os.environ.get("HOME"))

    endpoint = os.environ["WEKA_ENDPOINT_URL"]

    # Prefer explicit env creds (best for beaker / torchrun)
    key = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    token = os.getenv("WEKA_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")

    client_kwargs = {
        "endpoint_url": endpoint,
        # botocore often wants a region even for S3-compatible endpoints
        "region_name": os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1",
    }

    if key and secret:
        return s3fs.S3FileSystem(
            key=key,
            secret=secret,
            token=token,
            client_kwargs=client_kwargs,
        )

    # Fallback: use a profile (keeps local behavior)
    profile = os.getenv("WEKA_AWS_PROFILE", "WEKA")
    return s3fs.S3FileSystem(client_kwargs=client_kwargs, profile=profile)

@dataclass
class MixtureBuilder:
    sources: List[SourceInstance]
    max_tokens: int
    global_batch_size: int
    sequence_length: int
    seed: int
    dtype: NumpyDatasetDType
    processes: int = 1

    # cached_fs: dict[str, Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]] = field(
    #     default_factory=lambda: dict(
    #         s3=s3fs.S3FileSystem(),
    #         weka=_make_weka_fs(),
    #         gs=gcsfs.GCSFileSystem(),
    #     )
    # )
    cached_fs: dict[str, Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]] = field(
        default_factory=lambda: dict(
            s3=s3fs.S3FileSystem(),
            # weka=s3fs.S3FileSystem(
            #     client_kwargs={"endpoint_url": os.environ["WEKA_ENDPOINT_URL"]}, profile="WEKA"
            # ),
            gs=gcsfs.GCSFileSystem(),
        )
    )

    def build(self) -> SourceMixtureDatasetConfig:
        source_configs: List[SourceMixtureConfig] = []
        for source in self.sources:
            globs = [path for path in source.paths if "*" in path]
            paths = [path for path in source.paths if path not in globs]

            # Check if all paths have the same URL scheme
            schemes = {urlparse(path).scheme for path in paths + globs}
            if len(schemes) > 1:
                raise ValueError(f"All paths for source {source.name} must have the same scheme. Found: {schemes}")
            elif len(schemes) == 0:
                raise ValueError(f"No paths found for source {source.name}")

            scheme = schemes.pop()

            expanded = paths + expand_globs(self.cached_fs.get(scheme, self.cached_fs["s3"]), globs)

            if len(expanded) == 0:
                raise ValueError(f"No paths found for source {source.name}")

            source_configs.append(
                SourceMixtureConfig(
                    source_name=source.name,
                    paths=expanded,
                    target_ratio=source.ratio,
                    max_repetition_ratio=source.repetition_factor,
                )
            )

        source_list = SourceMixtureList(sources=source_configs)

        return SourceMixtureDatasetConfig(
            source_list=source_list,           # <--- Fix: Renamed from source_configs
            requested_tokens=self.max_tokens,  # <--- Fix: Renamed from max_tokens
            global_batch_size=self.global_batch_size, # <--- Fix: Added required arg
            seed=self.seed,
            processes=self.processes,
            # sequence_length and dtype removed: these belong in the .build() method call, not here
        )

        # return SourceMixtureDatasetConfig(
        #     source_configs=source_configs,
        #     max_tokens=self.max_tokens,
        #     sequence_length=self.sequence_length,
        #     seed=self.seed,
        #     dtype=self.dtype,
        #     processes=self.processes,
        # )

