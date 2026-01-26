import os
from dataclasses import dataclass, field
from typing import List, Union
from urllib.parse import urlparse

import gcsfs
import s3fs
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
)
from olmo_core.data.types import NumpyDatasetDType

from cookbook.aliases import SourceInstance
from cookbook.utils.data import expand_globs


@dataclass
class MixtureBuilder:
    sources: List[SourceInstance]
    max_tokens: int
    sequence_length: int
    seed: int
    dtype: NumpyDatasetDType
    processes: int = 1
    cached_fs: dict[str, Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]] = field(
        default_factory=lambda: dict(s3=s3fs.S3FileSystem())  # Only init S3 eagerly
    )

    def _get_fs(self, scheme: str) -> Union[s3fs.S3FileSystem, gcsfs.GCSFileSystem]:
        """Get filesystem for scheme, initializing lazily if needed."""
        if scheme not in self.cached_fs:
            if scheme == "weka":
                # profile="WEKA" reads endpoint_url from AWS config [profile WEKA]
                self.cached_fs[scheme] = s3fs.S3FileSystem(profile="WEKA")
            elif scheme == "gs":
                self.cached_fs[scheme] = gcsfs.GCSFileSystem()
            else:
                # Fall back to S3 for unknown schemes
                return self.cached_fs["s3"]
        return self.cached_fs[scheme]

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

            expanded = paths + expand_globs(self._get_fs(scheme), globs)

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

        return SourceMixtureDatasetConfig(
            source_configs=source_configs,
            max_tokens=self.max_tokens,
            sequence_length=self.sequence_length,
            seed=self.seed,
            dtype=self.dtype,
            processes=self.processes,
        )
