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


@dataclass
class MixtureBuilder:
    sources: List[SourceInstance]
    max_tokens: int
    global_batch_size: int
    sequence_length: int
    seed: int
    dtype: NumpyDatasetDType
    processes: int = 1
    cached_fs: dict[str, Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]] = field(
        default_factory=lambda: dict(
            s3=s3fs.S3FileSystem(),
            weka=s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": os.environ["WEKA_ENDPOINT_URL"]}, profile="WEKA"
            ),
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

