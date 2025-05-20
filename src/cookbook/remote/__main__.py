from typing import Any
import uuid
from .base import LocatedPath
from .gantry_launcher import GantryLauncher

from .gcp import GoogleCloudToken
from cookbook.cli.utils import PythonEnv


def copy_prefix(
    src_path: str,
    dst_path: str,
    *args: Any,
    **kwargs: Any,
):
    src_loc = LocatedPath.from_str(src_path)
    dst_loc = LocatedPath.from_str(dst_path)

    if src_loc.prot == "gs":
        if dst_loc.prot in ("weka", "file"):
            from .gcp import download_gcs_prefix

            download_gcs_prefix(src_loc.remote, dst_loc.local, *args, **kwargs)
        elif dst_loc.prot == "gs":
            raise ValueError("GCS -> GCS: not supported")
        elif dst_loc.prot == "s3":
            raise NotImplementedError("GCS -> S3: not implemented")

    elif src_loc.prot == "s3":
        raise NotImplementedError(f"S3 -> {dst_loc.prot.upper()}: not implemented")

    elif src_loc.prot in ("weka", "file"):
        if dst_loc.prot in ("weka", "file"):
            raise ValueError("Weka -> Weka: not supported")
        elif dst_loc.prot == "gs":
            from .gcp import upload_gcs_prefix

            upload_gcs_prefix(src_loc.local, dst_loc.remote, *args, **kwargs)
        elif dst_loc.prot == "s3":
            raise NotImplementedError("Weka -> S3: not implemented")

    else:
        raise ValueError(f"{src_loc.prot.upper()} -> {dst_loc.prot.upper()}: not recognized")


def main():
    import os
    import tempfile

    beaker_experiment_id = os.environ.get("BEAKER_EXPERIMENT_ID")

    if beaker_experiment_id:
        # fetch google cloud token
        gct = GoogleCloudToken.from_json(t) if (t := os.environ.get("GOOGLE_CLOUD_TOKEN")) else None

        # running in beaker
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, f"{uuid.uuid4()}.txt"), "w") as f:
                f.write("Hello, world!")

            return copy_prefix(
                src_path=tmp_dir,
                dst_path=f"gs://ai2-llm/temp/{beaker_experiment_id}",
                google_cloud_token=gct,
            )

    env = PythonEnv.create("test-env")

    bw = GantryLauncher(
        allow_dirty=True,
        budget="ai2/oe-data",
        cluster="ai2/ceres-cirrascale",
        dry_run=False,
        gpus=0,
        priority="high",
        preemptible=False,
        workspace="ai2/oe-data",
        env=env,
    )

    gct = GoogleCloudToken.make()
    bw.add_env_secret("GOOGLE_CLOUD_TOKEN", gct.to_json(), overwrite=True)

    bw.run(
        command="python -m cookbook.remote",
        description="Hello, world!",
    )


if __name__ == "__main__":
    main()
