#!/usr/bin/env python3
"""
Model merging execution for olmo-cookbook.

Handles both local and Beaker-based model merging execution.
"""

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

from cookbook.cli.utils import PythonEnv, find_repository_root, discover_weka_mount
from cookbook.constants import BEAKER_KNOWN_CLUSTERS

logger = logging.getLogger(__name__)


def run_merge_job(
    checkpoint_paths: List[str],
    output_dir: str,
    methods: List[str],
    alpha: float = 0.2,
    format: str = "auto",
    verbose: bool = False,
    use_beaker: bool = False,
    beaker_workspace: str = "ai2/oe-data",
    beaker_priority: str = "high", 
    beaker_cluster: str = "aus80g",
    beaker_budget: str = "ai2/oe-data",
    beaker_gpus: int = 1,
    beaker_image: str = "oe-eval-beaker/oe_eval_olmo3_auto",
    beaker_dry_run: bool = False,
    beaker_allow_dirty: bool = False,
):
    """
    Run model merging either locally or on Beaker.
    
    Args:
        checkpoint_paths: List of checkpoint paths to merge
        output_dir: Directory to save merged models
        methods: List of merging methods ['sma', 'wma', 'ema']
        alpha: Smoothing factor for EMA
        format: Output format ('safetensors', 'pytorch', 'auto')
        verbose: Enable verbose logging
        use_beaker: Whether to run on Beaker
        beaker_*: Beaker-specific configuration options
    """
    
    if use_beaker:
        return _run_merge_on_beaker(
            checkpoint_paths=checkpoint_paths,
            output_dir=output_dir,
            methods=methods,
            alpha=alpha,
            format=format,
            verbose=verbose,
            workspace=beaker_workspace,
            priority=beaker_priority,
            cluster=beaker_cluster,
            budget=beaker_budget,
            gpus=beaker_gpus,
            image=beaker_image,
            dry_run=beaker_dry_run,
            allow_dirty=beaker_allow_dirty,
        )
    else:
        return _run_merge_locally(
            checkpoint_paths=checkpoint_paths,
            output_dir=output_dir,
            methods=methods,
            alpha=alpha,
            format=format,
            verbose=verbose,
        )


def _run_merge_locally(
    checkpoint_paths: List[str],
    output_dir: str,
    methods: List[str],
    alpha: float,
    format: str,
    verbose: bool,
):
    from cookbook.model.merging import ModelMerger
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if len(checkpoint_paths) < 2:
        raise ValueError(f"Need at least 2 checkpoints for merging, got {len(checkpoint_paths)}")
    
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    if not methods:
        methods = ['sma', 'wma', 'ema']

    force_safetensors = None
    if format == "safetensors":
        force_safetensors = True
    elif format == "pytorch":
        force_safetensors = False
    # else: auto-detect (force_safetensors = None)
    
    merger = ModelMerger(
        checkpoint_paths=checkpoint_paths,
        output_dir=output_dir,
        alpha=alpha,
        force_safetensors=force_safetensors
    )
    
    logger.info(f"Starting local merge of {len(checkpoint_paths)} checkpoints")
    logger.info(f"Methods: {', '.join(methods)}")
    logger.info(f"Output directory: {output_dir}")
    
    merger.merge_models(methods=methods)
    
    logger.info("✅ Model merging completed successfully!")
    logger.info(f"📁 Output directory: {output_dir}")


def _run_merge_on_beaker(
    checkpoint_paths: List[str],
    output_dir: str,
    methods: List[str],
    alpha: float,
    format: str,
    verbose: bool,
    workspace: str,
    priority: str,
    cluster: str,
    budget: str,
    gpus: int,
    image: str,
    dry_run: bool,
    allow_dirty: bool,
):
    logger.info("Launching model merge job on Beaker")
    
    merge_cmd = _build_merge_command(
        checkpoint_paths=checkpoint_paths,
        output_dir=output_dir,
        methods=methods,
        alpha=alpha,
        format=format,
        verbose=verbose,
    )
    gantry_cmd, _ = _build_beaker_cli_command(
        merge_command=merge_cmd,
        workspace=workspace,
        priority=priority,
        cluster=cluster,
        budget=budget,
        gpus=gpus,
        image=image,
        allow_dirty=allow_dirty,
        checkpoint_paths=checkpoint_paths,
        output_dir=output_dir,
    )
    
    if dry_run:
        logger.info("Dry run mode - would run the following command:")
        logger.info(" ".join(gantry_cmd))
        return
    try:
        gantry_command_str = " ".join(gantry_cmd)
        logger.info(f"Running: {gantry_command_str}")
        
        env = PythonEnv.null()
        result = subprocess.run(
            shlex.split(gantry_command_str),
            capture_output=True,
            text=True,
            check=True,
            env=env.path()
        )
        logger.info("Gantry job submitted successfully")
        logger.info(f"Output: {result.stdout}")
        return "submitted"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to submit gantry job: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise


def _build_merge_command(
    checkpoint_paths: List[str],
    output_dir: str,
    methods: List[str],
    alpha: float,
    format: str,
    verbose: bool,
) -> List[str]:
    cmd = [
        "pip install uv && uv pip install .[all] --system &&",
        "olmo-cookbook-merge merge"
    ]

    cmd.extend(checkpoint_paths)
    cmd.extend(["--output-dir", output_dir])
    
    if methods:
        for method in methods:
            cmd.extend(["--methods", method])
    
    cmd.extend(["--alpha", str(alpha)])
    cmd.extend(["--format", format])
    
    if verbose:
        cmd.append("--verbose")
    
    return cmd


def _build_beaker_cli_command(
    merge_command: List[str],
    workspace: str,
    priority: str,
    cluster: str,
    budget: str,
    gpus: int,
    image: str,
    allow_dirty: bool,
    checkpoint_paths: List[str],
    output_dir: str,
) -> tuple[List[str], str]:
    from cookbook.cli.utils import install_beaker_py
    env = PythonEnv.null()
    install_beaker_py(env=env)
    try:
        repo_root = find_repository_root()
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        is_dirty = bool(result.stdout.strip())
        
        if is_dirty and not allow_dirty:
            raise RuntimeError("Repository has uncommitted changes. Use --beaker-allow-dirty to override.")
            
    except Exception as e:
        if not allow_dirty:
            raise RuntimeError(f"Could not determine git state: {e}")
        logger.warning(f"Could not determine git state: {e}")

    weka_mounts = [
        mount
        for mount in (
            *[discover_weka_mount(path) for path in checkpoint_paths],
            discover_weka_mount(output_dir),
        )
        if mount is not None
    ]

    gantry_flags = []

    for weka_path in set(weka_mounts):
        gantry_flags.append(f"--weka {weka_path}:/{weka_path}")

    for cluster_name in BEAKER_KNOWN_CLUSTERS.get(cluster, [cluster]):
        gantry_flags.append(f"--cluster {cluster_name}")

    remote_command_str = " ".join(merge_command)

    import time
    timestamp = int(time.time())
    gantry_command = [
        "gantry run",
        f"--description 'Model merging job {timestamp}'",
        ("--allow-dirty" if allow_dirty else ""),
        "--no-python",
        f"--workspace {workspace}",
        f"--priority {priority}",
        f"--gpus 0",
        f"--budget {budget}",
        "--yes",
        " ".join(gantry_flags),
        f"-- /bin/bash -c '{remote_command_str}'",
    ]
    
    return gantry_command, ""