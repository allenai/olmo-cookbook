from typing import Any
from .base import LocatedPath


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


def copy_prefix_remote(
    beaker_allow_dirty: bool,
    beaker_budget: str,
    beaker_cluster: str,
    beaker_dry_run: bool,
    beaker_gpus: int,
    beaker_priority: str,
    beaker_workspace: str,
    beaker_preemptible: bool,
    huggingface_output_dir: Optional[str],
    huggingface_output_suffix: str,
    huggingface_token: Optional[str],
    huggingface_tokenizer: Optional[str],
    input_dir: str,
    olmo2_commit_hash: str,
    olmo_type: str,
    olmoe_commit_hash: str,
    olmo_core_commit_hash: str,
    olmo_core_v2_commit_hash: str,
    huggingface_transformers_commit_hash: str,
    unsharded_output_dir: Optional[str],
    unsharded_output_suffix: str,
    use_beaker: bool,
    use_system_python: bool,
    python_venv_name: str,
    python_venv_force: bool,
    max_sequence_length: Optional[int] = None,
):
    env = (
        PythonEnv.create(name=python_venv_name, force=python_venv_force)
        if not use_system_python
        else PythonEnv.null()
    )
