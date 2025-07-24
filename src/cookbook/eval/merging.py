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
    """Run model merging on Beaker."""
    try:
        import beaker
        from beaker import Beaker
    except ImportError:
        raise ImportError("beaker-py must be installed to use Beaker functionality")
    
    logger.info("🚀 Launching model merge job on Beaker")
    
    # Create the merge command
    merge_cmd = _build_merge_command(
        checkpoint_paths=checkpoint_paths,
        output_dir=output_dir,
        methods=methods,
        alpha=alpha,
        format=format,
        verbose=verbose,
    )
    
    # Build Beaker job specification
    job_spec = _build_beaker_job_spec(
        command=merge_cmd,
        workspace=workspace,
        priority=priority,
        cluster=cluster,
        budget=budget,
        gpus=gpus,
        image=image,
        allow_dirty=allow_dirty,
    )
    
    if dry_run:
        logger.info("🧪 Dry run mode - would submit the following job:")
        logger.info(json.dumps(job_spec, indent=2))
        return
    
    # Submit job to Beaker
    client = Beaker.from_env(default_workspace=workspace)
    
    try:
        job = client.job.submit(
            spec=beaker.JobSpec.from_dict(job_spec),
            name=f"merge-{'-'.join(Path(p).name for p in checkpoint_paths[:3])}"[:50]
        )
        
        logger.info(f"✅ Beaker job submitted: {job.id}")
        logger.info(f"🔗 View job at: https://beaker.org/job/{job.id}")
        
        return job.id
        
    except Exception as e:
        logger.error(f"❌ Failed to submit Beaker job: {e}")
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
    
    cmd = [
        "python", "-m", "cookbook.cli.merge", "merge"
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


def _build_beaker_job_spec(
    command: List[str],
    workspace: str,
    priority: str,
    cluster: str,
    budget: str,
    gpus: int,
    image: str,
    allow_dirty: bool,
) -> dict:
    """Build Beaker job specification."""
    
    # Get repository information
    try:
        repo_root = find_repository_root()
        
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result.stdout.strip()
        
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
        logger.warning(f"Could not determine git state: {e}")
        commit_hash = "unknown"
    
    # Build job specification
    spec = {
        "version": "v2",
        "description": f"Model merging job - commit {commit_hash[:8]}",
        "tasks": [
            {
                "name": "merge",
                "image": {"beaker": image},
                "command": command,
                "resources": {
                    "gpuCount": gpus,
                },
                "context": {
                    "cluster": cluster,
                    "priority": priority,
                },
                "constraints": {
                    "cluster": [cluster]
                },
                "datasets": [
                    {
                        "mountPath": "/oe-training-default",
                        "source": {"weka": "oe-training-default"}
                    }
                ],
                "envVars": [
                    {
                        "name": "WANDB_DISABLED", 
                        "value": "true"
                    }
                ]
            }
        ],
        "budget": budget,
    }
    
    return spec