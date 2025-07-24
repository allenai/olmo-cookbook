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

from cookbook.cli.utils import PythonEnv, find_repository_root
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
    """Run model merging locally."""
    from cookbook.model.merging import ModelMerger
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if len(checkpoint_paths) < 2:
        raise ValueError(f"Need at least 2 checkpoints for merging, got {len(checkpoint_paths)}")
    
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    # Default to all methods if none specified
    if not methods:
        methods = ['sma', 'wma', 'ema']
    
    # Determine force_safetensors parameter
    force_safetensors = None
    if format == "safetensors":
        force_safetensors = True
    elif format == "pytorch":
        force_safetensors = False
    # else: auto-detect (force_safetensors = None)
    
    # Create merger and run
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
    """Run model merging on Beaker using beaker CLI."""
    logger.info("🚀 Launching model merge job on Beaker")
    
    # Create the merge command that will run inside Beaker
    merge_cmd = _build_merge_command(
        checkpoint_paths=checkpoint_paths,
        output_dir=output_dir,
        methods=methods,
        alpha=alpha,
        format=format,
        verbose=verbose,
    )
    
    # Build the gantry command
    gantry_cmd, _ = _build_beaker_cli_command(
        merge_command=merge_cmd,
        workspace=workspace,
        priority=priority,
        cluster=cluster,
        budget=budget,
        gpus=gpus,
        image=image,
        allow_dirty=allow_dirty,
    )
    
    if dry_run:
        logger.info("🧪 Dry run mode - would run the following command:")
        logger.info(" ".join(gantry_cmd))
        return
    
    # Submit job using gantry
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
        
        logger.info("✅ Gantry job submitted successfully")
        logger.info(f"Output: {result.stdout}")
        return "submitted"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to submit gantry job: {e}")
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
    """Build the command to run merge locally within Beaker."""
    
    # Build command following the same pattern as conversion.py
    cmd = [
        "pip install uv && uv pip install . --system &&",
        "olmo-cookbook-merge merge"
    ]
    
    # Add checkpoint paths
    cmd.extend(checkpoint_paths)
    
    # Add options
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
) -> tuple[List[str], str]:
    """Build gantry command to submit the job, following conversion.py pattern."""
    from cookbook.cli.utils import install_beaker_py
    
    # Install beaker and gantry clients
    env = PythonEnv.null()
    install_beaker_py(env=env)
    
    # Check git state
    try:
        repo_root = find_repository_root()
        
        # Check if repository is dirty
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
    
    # Build gantry flags
    gantry_flags = []
    
    # Add weka mount for /oe-training-default
    gantry_flags.append("--weka oe-training-default:/oe-training-default")
    
    # Add cluster constraints
    for cluster_name in BEAKER_KNOWN_CLUSTERS.get(cluster, [cluster]):
        gantry_flags.append(f"--cluster {cluster_name}")
    
    # Build remote command string
    remote_command_str = " ".join(merge_command)
    
    # Build gantry command
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