import argparse
import os
import tempfile
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
    parser = argparse.ArgumentParser("Move prefixes between storage systems")
    parser.add_argument("src_path", type=str, help="Source path")
    parser.add_argument("dst_path", type=str, help="Destination path")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--google-cloud-token", type=str, default=None, help="Google Cloud token")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty operations")
    parser.add_argument("--budget", type=str, default="ai2/oe-data", help="Budget")
    parser.add_argument("--cluster", type=str, default="aus", help="Clusters to run on")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs")
    parser.add_argument("--priority", type=str, default="high", help="Priority")
    parser.add_argument("--preemptible", action="store_true", help="Preemptible")
    parser.add_argument("--workspace", type=str, default="ai2/oe-data", help="Workspace")
    args = parser.parse_args()

    if (beaker_experiment_id := os.environ.get("BEAKER_EXPERIMENT_ID")):
        # running on beaker, do the actual work
        gct = GoogleCloudToken.from_json(t) if (t := os.environ.get("GOOGLE_CLOUD_TOKEN")) else None
        copy_prefix(
            src_path=args.src_path,
            dst_path=args.dst_path,
            google_cloud_token=gct,
            num_workers=args.num_workers,
        )

    else:
        # running locally, submit to beaker
        env = PythonEnv.create("copy-prefix")
        bw = GantryLauncher(
            allow_dirty=args.allow_dirty,
            budget=args.budget,
            cluster=args.cluster,
            dry_run=args.dry_run,
            gpus=args.gpus,
            priority=args.priority,
            preemptible=args.preemptible,
            workspace=args.workspace,
            env=env,
        )

        gct = GoogleCloudToken.make()
        bw.add_env_secret("GOOGLE_CLOUD_TOKEN", gct.to_json(), overwrite=True)

        bw.run(
            command=f"python -m cookbook.remote '{args.src_path}' '{args.dst_path}'",
            description=f"Copying {args.src_path} to {args.dst_path}",
        )


if __name__ == "__main__":
    main()
