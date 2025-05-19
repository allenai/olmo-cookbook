from typing import Any
from .base import LocatedPath
from .beakerfy import BeakerWrapper

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
        from .gcp import download_gcs_prefix
        if dst_loc.prot != "weka":
            raise ValueError("Cannot copy from GCS to non-Weka destination")
        else:
            download_gcs_prefix(src_loc.to_str(), dst_loc.to_str(), *args, **kwargs)

    elif src_loc.prot == "s3":
        raise NotImplementedError("S3 copy not implemented")

    elif src_loc.prot == "weka":
        if dst_loc.prot == "weka":
            raise ValueError("Cannot copy from Weka to Weka")
        elif dst_loc.prot == "gs":
            from .gcp import upload_gcs_prefix
            upload_gcs_prefix(src_loc.to_str(), dst_loc.to_str(), *args, **kwargs)
        elif dst_loc.prot == "s3":
            raise NotImplementedError("S3 copy not implemented")

    raise ValueError(f"Cannot copy from {src_loc.prot} to {dst_loc.prot} destination")


def main():
    import os
    import tempfile

    beaker_experiment_id = os.environ.get("BEAKER_EXPERIMENT_ID")

    if beaker_experiment_id:
        cloud_token = GoogleCloudToken.from_json(os.environ["GOOGLE_CLOUD_TOKEN"])
        google_cloud_token = cloud_token.apply()

        # running in beaker
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "output.txt"), "w") as f:
                f.write("Hello, world!")

            return copy_prefix(
                src_path=tmp_dir,
                dst_path=f"gs://ai2-llm/temp/{beaker_experiment_id}",
                google_cloud_token=google_cloud_token,
            )

    env = PythonEnv.create('test-env')

    bw = BeakerWrapper(
        allow_dirty=True,
        budget="ai2/oe-data",
        cluster='ai2/ceres-cirrascale',
        dry_run=False,
        gpus=0,
        priority="high",
        preemptible=False,
        workspace="ai2/oe-data",
        env=env,
    )

    gct = GoogleCloudToken.make()
    bw.add_env_secret("GOOGLE_CLOUD_TOKEN", gct.token, overwrite=True)

    bw.run(
        command="python -m cookbook.remote",
        description="Hello, world!",
    )


if __name__ == "__main__":
    main()
