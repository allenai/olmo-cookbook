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
    
    # Build the beaker CLI command
    beaker_cmd, spec_file = _build_beaker_cli_command(
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
        logger.info(" ".join(beaker_cmd))
        logger.info(f"Using spec file: {spec_file}")
        # Clean up spec file
        os.unlink(spec_file)
        return
    
    # Submit job using beaker CLI
    try:
        logger.info(f"Running: {' '.join(beaker_cmd)}")
        result = subprocess.run(
            beaker_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract experiment ID from output
        output_lines = result.stdout.strip().split('\n')
        experiment_id = None
        for line in output_lines:
            if 'experiment' in line.lower() and ('created' in line.lower() or 'submitted' in line.lower()):
                # Try to extract experiment ID from output
                import re
                match = re.search(r'[a-f0-9]{8}', line)
                if match:
                    experiment_id = match.group(0)
                    break
        
        if experiment_id:
            logger.info(f"✅ Beaker experiment created: {experiment_id}")
            logger.info(f"🔗 View experiment at: https://beaker.org/ex/{experiment_id}")
            return experiment_id
        else:
            logger.info("✅ Beaker experiment submitted successfully")
            logger.info(f"Output: {result.stdout}")
            return "submitted"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to submit Beaker experiment: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    finally:
        # Clean up spec file
        if os.path.exists(spec_file):
            os.unlink(spec_file)


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
        "python", "-c", 
        "import sys; sys.path.insert(0, '/workdir/src'); import cookbook.cli.merge; cookbook.cli.merge.cli()",
        "merge"
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
    """Build beaker CLI command and spec file to submit the job."""
    import tempfile
    import yaml
    
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
    
    # Map cluster name to actual Beaker cluster names
    if cluster in BEAKER_KNOWN_CLUSTERS:
        actual_clusters = BEAKER_KNOWN_CLUSTERS[cluster]
    else:
        # Assume it's already a full cluster name
        actual_clusters = [cluster]
    
    # Create experiment spec
    spec = {
        "version": "v2",
        "description": "Model merging job",
        "budget": budget,
        "tasks": [
            {
                "name": "merge",
                "image": {"beaker": image},
                "command": merge_command,
                "resources": {
                    "gpuCount": gpus,
                },
                "context": {
                    "priority": priority,
                },
                "constraints": {
                    "cluster": actual_clusters
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
        ]
    }
    
    # Create temporary spec file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec, f, default_flow_style=False)
        spec_file = f.name
    
    # Build beaker command with unique timestamp
    import time
    timestamp = int(time.time())
    cmd = [
        "beaker", "experiment", "create",
        spec_file,
        "--workspace", workspace,
        "--name", f"merge-{Path(merge_command[4]).name}-{Path(merge_command[5]).name}-{timestamp}"[:50]
    ]
    
    return cmd, spec_file