import argparse
import os
from hashlib import md5
from typing import Any

from cookbook.cli.utils import PythonEnv

from .base import BaseAuthentication, LocatedPath
from .gantry_launcher import GantryLauncher


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
        if dst_loc.prot in ("weka", "file"):
            from .aws import download_s3_prefix

            download_s3_prefix(src_loc.remote, dst_loc.local, *args, **kwargs)
        elif dst_loc.prot == "s3":
            raise ValueError("S3 -> S3: not supported")
        elif dst_loc.prot == "gs":
            raise NotImplementedError("S3 -> GCS: not implemented")

    elif src_loc.prot in ("weka", "file"):
        if dst_loc.prot in ("weka", "file"):
            raise ValueError("Weka -> Weka: not supported")
        elif dst_loc.prot == "gs":
            from .gcp import upload_gcs_prefix

            upload_gcs_prefix(src_loc.local, dst_loc.remote, *args, **kwargs)
        elif dst_loc.prot == "s3":
            from .aws import upload_s3_prefix

            upload_s3_prefix(src_loc.local, dst_loc.remote, *args, **kwargs)

    else:
        raise ValueError(f"{src_loc.prot.upper()} -> {dst_loc.prot.upper()}: not recognized")


def push_credentials(gantry_launcher: GantryLauncher, *paths: str):
    for path in paths:
        loc = LocatedPath.from_str(path)
        if loc.prot == "gs":
            from .gcp import GoogleCloudToken

            gct = GoogleCloudToken.make()
            return gantry_launcher.add_env_secret(f"COOKBOOK_AUTH_{loc.hash[:6]}", gct.to_json(), overwrite=True)
        elif loc.prot == "s3":
            from .aws import AwsCredentials

            aws_creds = AwsCredentials.make()
            return gantry_launcher.add_env_secret(
                f"COOKBOOK_AUTH_{loc.hash[:6]}", aws_creds.to_json(), overwrite=True
            )


def pull_credentials(*paths: str) -> BaseAuthentication | None:
    for path in paths:
        loc = LocatedPath.from_str(path)
        if loc.prot == "gs":
            from .gcp import GoogleCloudToken

            return GoogleCloudToken.from_json(os.environ[f"COOKBOOK_AUTH_{loc.hash[:6]}"])
        elif loc.prot == "s3":
            from .aws import AwsCredentials

            return AwsCredentials.from_json(os.environ[f"COOKBOOK_AUTH_{loc.hash[:6]}"])
    return None


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
    parser.add_argument("--local-only", action="store_true", help="Local only")
    parser.add_argument(
        "--credentials_env_name", type=str, default="COOKBOOK_REMOTE_CREDENTIALS", help="Credentials env name"
    )
    args = parser.parse_args()

    if os.environ.get("BEAKER_EXPERIMENT_ID") or args.local_only:

        # only pull credentials if running on beaker
        creds = pull_credentials(args.src_path, args.dst_path) if not args.local_only else None

        copy_prefix(
            src_path=args.src_path,
            dst_path=args.dst_path,
            credentials=creds,
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

        # adds mount if necessary
        bw.add_mount(args.src_path)
        bw.add_mount(args.dst_path)

        push_credentials(bw, args.src_path, args.dst_path)

        bw.run(
            command=f"python -m cookbook.remote '{args.src_path}' '{args.dst_path}'",
            description=f"Copying {args.src_path} to {args.dst_path}",
        )


if __name__ == "__main__":
    main()
