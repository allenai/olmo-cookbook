import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import boto3
from tqdm import tqdm

from ..cli.utils import get_aws_access_key_id, get_aws_secret_access_key
from .base import AuthenticationError, BaseAuthentication

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client  # pyright: ignore


@dataclass(frozen=True)
class AwsCredentials(BaseAuthentication):
    access_key_id: str
    secret_access_key: str

    @classmethod
    def make(cls) -> "AwsCredentials":
        access_key_id = get_aws_access_key_id()
        secret_access_key = get_aws_secret_access_key()
        if access_key_id is None or secret_access_key is None:
            raise AuthenticationError("No AWS credentials found")
        return cls(access_key_id=access_key_id, secret_access_key=secret_access_key)

    def apply(self) -> boto3.Session:
        """Apply the credentials so that it can be used for remote operations."""
        return boto3.Session(aws_access_key_id=self.access_key_id, aws_secret_access_key=self.secret_access_key)


def list_objects_with_paginator(bucket_name: str, prefix: str, client: "S3Client"):
    """
    List all objects in an S3 bucket using boto3's paginator.
    This automatically handles pagination for you.
    """
    # Create a paginator for list_objects_v2
    paginator = client.get_paginator("list_objects_v2")

    # Configure the pagination parameters
    page_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
        PaginationConfig={
            "MaxItems": None,  # Return all items
            "PageSize": 1000,  # Number of items per page (max 1000)
        },
    )

    # Iterate through all pages
    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                yield bucket_name, obj["Key"]


def download_s3_prefix(
    remote_path: str,
    local_path: str | Path,
    session: boto3.Session | None = None,
    num_workers: int | None = None,
    credentials: AwsCredentials | None = None,
):
    protocol, bucket_name, prefix = (p := urlparse(remote_path)).scheme, p.netloc, p.path.lstrip("/")
    assert protocol.startswith("s3"), "Only S3 and S3A protocols are supported"

    client = (credentials.apply() if credentials else (session or boto3.Session())).client("s3")

    # Create a local directory if it doesn't exist
    local_path = Path(local_path)
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for bucket, key in list_objects_with_paginator(bucket_name, prefix, client):
            local_file_path = local_path / Path(key).relative_to(Path(prefix))

            def _download_file(
                _bucket: str,
                _key: str,
                _local_file_path: Path,
                _client: "S3Client",
            ):
                _local_file_path.parent.mkdir(parents=True, exist_ok=True)
                _client.download_file(_bucket, _key, str(_local_file_path))

            futures.append(
                executor.submit(
                    _download_file,
                    _bucket=bucket,
                    _key=key,
                    _local_file_path=local_file_path,
                    _client=client,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading prefix"):
            try:
                future.result()
            except Exception as e:
                for future_to_cancel in futures:
                    future_to_cancel.cancel()
                raise e


def upload_s3_prefix(
    local_path: str | Path,
    remote_path: str,
    session: boto3.Session | None = None,
    num_workers: int | None = None,
    credentials: AwsCredentials | None = None,
):
    protocol, bucket_name, prefix = (p := urlparse(remote_path)).scheme, p.netloc, p.path.lstrip("/")
    assert protocol.startswith("s3"), "Only S3 and S3A protocols are supported"

    client = (credentials.apply() if credentials else (session or boto3.Session())).client("s3")
    local_path = Path(local_path).absolute()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for dp, _, files in os.walk(str(local_path)):
            for fp_str in files:
                fp = Path(dp) / fp_str
                if not fp.is_file():
                    continue

                def _upload_file(
                    _fp: Path,
                    _bucket: str,
                    _key: str,
                    _client: "S3Client",
                ):
                    _client.upload_file(str(_fp), _bucket, _key)

                futures.append(
                    executor.submit(
                        _upload_file,
                        _fp=fp,
                        _bucket=bucket_name,
                        _key=f"{prefix}/{fp.relative_to(local_path)}",
                        _client=client,
                    )
                )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading prefix"):
            try:
                future.result()
            except Exception as e:
                for future_to_cancel in futures:
                    future_to_cancel.cancel()
                raise e
