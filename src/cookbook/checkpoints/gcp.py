import json
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


from google.auth import default
from google.auth.transport.requests import Request
from google.auth.exceptions import DefaultCredentialsError
import datetime

from google.oauth2.credentials import Credentials
from google.cloud import storage


from cookbook.cli.utils import PythonEnv


class ShortLivedToken(NamedTuple):
    token: str
    expiry: datetime.datetime | None

    @classmethod
    def from_json(cls, obj: str | dict[str, str]):
        obj = json.loads(obj) if isinstance(obj, str) else obj
        assert isinstance(obj, dict), "Invalid JSON object"
        expiry = datetime.datetime.fromisoformat(expiry_str) if (expiry_str := obj.get("expiry", None)) else None
        return cls(token=obj["token"], expiry=expiry)

    @classmethod
    def to_json(cls, obj: "ShortLivedToken") -> str:
        return json.dumps({"token": obj.token, "expiry": obj.expiry.isoformat() if obj.expiry else None})

    @classmethod
    def make(cls) -> "ShortLivedToken":
        """Generate short-lived token for GCS access."""

        # Get credentials from your environment (from gcloud auth login)
        try:
            credentials, _ = default()
        except DefaultCredentialsError as e:
            raise RuntimeError("No credentials found. Please run `gcloud auth login`") from e

        # Make sure the credentials are valid
        if not credentials.valid:   # pyright: ignore
            credentials.refresh(Request())  # pyright: ignore

        # Get the access token with expiration info
        token = credentials.token  # pyright: ignore
        expiry = credentials.expiry  # pyright: ignore

        return cls(token=token, expiry=expiry)

    def use_token_for_gcs(self) -> storage.Client:

        if self.expiry is not None and datetime.datetime.now() > self.expiry:
            raise RuntimeError("Token expired!")

        # Create credentials object from the token
        credentials = Credentials(self.token)

        # Use the credentials with the storage client
        client = storage.Client(credentials=credentials)

        return client


def download_gcs_prefix(
    remote_path: str,
    local_path: str | Path,
    client: storage.Client | None = None,
    num_workers: int | None = None,
    expiration_time: datetime.datetime | None = None,
):
    protocol, bucket_name, prefix = (p := urlparse(remote_path)).scheme, p.netloc, p.path.lstrip('/')
    assert protocol in ("gs", "gcs"), "Only GCS and GS protocols are supported"

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
                    _expiration_time=expiration_time,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading prefix"):
            try:
                future.result()
            except Exception as e:
                for future_to_cancel in futures:
                    future_to_cancel.cancel()
                raise e


def run_on_beaker

if __name__ == "__main__":
    # download_gcs_prefix(
    #     remote_path="gs://ai2-llm/checkpoints/lucas/sigdig-split-on-dolmino-1124-1b-5xC-38a22f26/step53971",
    #     local_path="/tmp/test",
    # )

    print(get_short_lived_token())
