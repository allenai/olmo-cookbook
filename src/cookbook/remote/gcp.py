from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base import BaseAuthentication, AuthenticationError, JSON_VALID_TYPES
from google.auth import default
from google.auth.transport.requests import Request
import datetime

from google.oauth2.credentials import Credentials
from google.cloud import storage


@dataclass(frozen=True)
class GoogleCloudToken(BaseAuthentication):
    token: str
    expiry: datetime.datetime | None

    @classmethod
    def from_dict(cls, obj: dict[str, JSON_VALID_TYPES]) -> "GoogleCloudToken":
        parsed_obj = {
            "token": obj['token'],
            "expiry": datetime.datetime.fromisoformat(e) if isinstance(e := obj.get("expiry", None), str) else e
        }
        return super().from_dict(parsed_obj)

    def to_dict(self) -> dict[str, JSON_VALID_TYPES]:
        obj = {
            "token": self.token,
            "expiry": self.expiry.isoformat() if self.expiry else None
        }
        return obj

    @classmethod
    def make(cls) -> "GoogleCloudToken":
        """Generate short-lived token for GCS access."""
        credentials, _ = default()
        if not credentials.valid:   # pyright: ignore
            credentials.refresh(Request())  # pyright: ignore

        return cls(token=credentials.token, expiry=credentials.expiry)  # pyright: ignore

    def apply(self) -> storage.Client:
        """Apply the credentials so that it can be used for remote operations."""

        if self.expiry is not None and datetime.datetime.now() > self.expiry:
            raise AuthenticationError("Token expired!")

        credentials = Credentials(self.token)
        return storage.Client(credentials=credentials)


def download_gcs_prefix(
    remote_path: str,
    local_path: str | Path,
    client: storage.Client | None = None,
    num_workers: int | None = None,
    google_cloud_token: GoogleCloudToken | None = None,
):
    protocol, bucket_name, prefix = (p := urlparse(remote_path)).scheme, p.netloc, p.path.lstrip('/')
    assert protocol in ("gs", "gcs"), "Only GCS and GS protocols are supported"

    if google_cloud_token is not None:
        client = google_cloud_token.apply()

    client = client or storage.Client()
    local_path = Path(local_path)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for blob in blobs:
            local_file_path = local_path / Path(blob.name).relative_to(Path(prefix))

            def _download_file(
                _blob: storage.Blob,
                _local_file_path: Path,
                _expiration_time: datetime.datetime | None,
            ):
                if _expiration_time is not None and datetime.datetime.now() > _expiration_time:
                    raise RuntimeError("Token expired!")

                _local_file_path.parent.mkdir(parents=True, exist_ok=True)
                _blob.download_to_filename(str(_local_file_path))

            futures.append(
                executor.submit(
                    _download_file,
                    _blob=blob,
                    _local_file_path=local_file_path,
                    _expiration_time=google_cloud_token.expiry if google_cloud_token else None,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading prefix"):
            try:
                future.result()
            except Exception as e:
                for future_to_cancel in futures:
                    future_to_cancel.cancel()
                raise e


def upload_gcs_prefix(
    local_path: str | Path,
    remote_path: str,
    client: storage.Client | None = None,
    num_workers: int | None = None,
    google_cloud_token: GoogleCloudToken | None = None,
):
    protocol, bucket_name, prefix = (p := urlparse(remote_path)).scheme, p.netloc, p.path.lstrip('/')
    assert protocol in ("gs", "gcs"), "Only GCS and GS protocols are supported"

    client = client or storage.Client()
    local_path = Path(local_path).absolute()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for dp, _, files in local_path.walk():
            for fp_str in files:
                fp = local_path / dp / Path(fp_str)
                if not fp.is_file():
                    continue

                bucket = client.bucket(bucket_name)
                blob = bucket.blob(f"{prefix}/{fp.relative_to(local_path)}")

                def _upload_file(
                    _fp: Path,
                    _blob: storage.Blob,
                    _expiration_time: datetime.datetime | None,
                ):
                    if _expiration_time is not None and datetime.datetime.now() > _expiration_time:
                        raise RuntimeError("Token expired!")

                    _blob.upload_from_filename(str(_fp))

                futures.append(
                    executor.submit(
                        _upload_file,
                        _fp=fp,
                        _blob=blob,
                        _expiration_time=token.expiry if token else None,
                    )
                )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading prefix"):
            try:
                future.result()
            except Exception as e:
                for future_to_cancel in futures:
                    future_to_cancel.cancel()
                raise e
