import click
import logging
from pathlib import Path

import glob

from olmo_core.utils import prepare_cli_environment

logger = logging.getLogger(__name__)

@click.group()
def cli():
    prepare_cli_environment()

@cli.command()
@click.argument('checkpoint_paths', nargs=-1, required=True)
@click.option('--output-dir', '-o', required=True, 
              help='Output directory for merged models')
@click.option('--methods', '-m', multiple=True, 
              type=click.Choice(['sma', 'wma', 'ema']), 
              help='Merging methods to use. Can specify multiple times.')
@click.option('--alpha', type=float, default=0.2, show_default=True,
              help='Smoothing factor for EMA (must be between 0 and 1)')
@click.option('--format', type=click.Choice(['safetensors', 'pytorch', 'auto']), 
              default='auto', show_default=True,
              help='Output format: safetensors, pytorch, or auto-detect')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.option('-b', '--use-beaker', is_flag=True, help='Use Beaker to run merge job')
@click.option('--beaker-workspace', type=str, default='ai2/oe-data', help='Beaker workspace')
@click.option('--beaker-priority', type=str, default='high', help='Beaker priority')
@click.option('--beaker-cluster', type=str, default='aus80g', help='Beaker cluster')
@click.option('--beaker-budget', type=str, default='ai2/oe-data', help='Beaker budget')
@click.option('--beaker-gpus', type=int, default=1, help='Number of GPUs for Beaker')
@click.option('--beaker-image', type=str, default='oe-eval-beaker/oe_eval_olmo3_auto', 
              help='Beaker image to use')
@click.option('--beaker-dry-run', is_flag=True, help='Dry run for Beaker')
@click.option('--beaker-allow-dirty', is_flag=True, help='Allow dirty workspace for Beaker')
def merge(checkpoint_paths: tuple, output_dir: str, methods: tuple, alpha: float, 
          format: str, verbose: bool, use_beaker: bool, beaker_workspace: str,
          beaker_priority: str, beaker_cluster: str, beaker_budget: str,
          beaker_gpus: int, beaker_image: str, beaker_dry_run: bool, 
          beaker_allow_dirty: bool):
    """
    Merge model checkpoints using various averaging strategies.
    
    CHECKPOINT_PATHS: Space-separated list of checkpoint paths (can be GCP/Weka/local paths)
    
    Examples:
    
    \b
    # Merge 3 checkpoints using all methods
    olmo-cookbook-merge merge /path/to/checkpoint1 /path/to/checkpoint2 /path/to/checkpoint3 \\
        --output-dir /path/to/output
    
    \b
    # Merge using only EMA with custom alpha
    olmo-cookbook-merge merge /path/to/checkpoint1 /path/to/checkpoint2 \\
        --output-dir /path/to/output --methods ema --alpha 0.3
    
    \b
    # Merge GCP checkpoints with specific format
    olmo-cookbook-merge merge gs://bucket/checkpoint1 gs://bucket/checkpoint2 \\
        --output-dir /local/output --format safetensors --verbose
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if len(checkpoint_paths) < 2:
        raise click.BadParameter(f"Need at least 2 checkpoints for merging, got {len(checkpoint_paths)}")
    
    if not 0 < alpha < 1:
        raise click.BadParameter(f"Alpha must be between 0 and 1, got {alpha}")
    
    if not methods:
        methods = ['sma', 'wma', 'ema']
    
    force_safetensors = None
    if format == "safetensors":
        force_safetensors = True
    elif format == "pytorch":
        force_safetensors = False
    # else: auto-detect (force_safetensors = None)
    
    try:
        from cookbook.eval.merging import run_merge_job
        result = run_merge_job(
            checkpoint_paths=list(checkpoint_paths),
            output_dir=output_dir,
            methods=list(methods) if methods else None,
            alpha=alpha,
            format=format,
            verbose=verbose,
            use_beaker=use_beaker,
            beaker_workspace=beaker_workspace,
            beaker_priority=beaker_priority,
            beaker_cluster=beaker_cluster,
            beaker_budget=beaker_budget,
            beaker_gpus=beaker_gpus,
            beaker_image=beaker_image,
            beaker_dry_run=beaker_dry_run,
            beaker_allow_dirty=beaker_allow_dirty,
        )
        
        if use_beaker and not beaker_dry_run:
            click.echo(f"Beaker job submitted: {result}")
        elif not use_beaker:
            click.echo("Model merging completed successfully!")
            click.echo(f"Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Merging failed: {str(e)}")
        raise click.ClickException(f"Merging failed: {str(e)}")

@cli.command()
@click.argument('checkpoint_path')
def validate(checkpoint_path: str):
    path = Path(checkpoint_path)
    
    if not path.exists():
        click.echo(f"Path does not exist: {checkpoint_path}")
        return
    
    if not path.is_dir():
        click.echo(f"Path is not a directory: {checkpoint_path}")
        return

    has_model = (
        (path / "pytorch_model.bin").exists() or
        (path / "model.safetensors").exists() or
        len(glob.glob(str(path / "pytorch_model-*.bin"))) > 0 or
        len(glob.glob(str(path / "model-*-of-*.safetensors"))) > 0
    )
    
    if not has_model:
        click.echo(f"No model files found in {checkpoint_path}")
        click.echo("Expected one of: model.safetensors, pytorch_model.bin, pytorch_model-*.bin, model-*-of-*.safetensors")
        return

    config_files = ["config.json", "tokenizer_config.json"]
    missing_configs = []
    
    for config_file in config_files:
        if not (path / config_file).exists():
            missing_configs.append(config_file)
    
    click.echo(f"Valid checkpoint found at: {checkpoint_path}")
    model_files = []
    if (path / "model.safetensors").exists():
        model_files.append("model.safetensors")
    if (path / "pytorch_model.bin").exists():
        model_files.append("pytorch_model.bin")
    
    sharded_safetensors = glob.glob(str(path / "model-*-of-*.safetensors"))
    if sharded_safetensors:
        model_files.append(f"{len(sharded_safetensors)} sharded safetensors files")
    
    sharded_pytorch = glob.glob(str(path / "pytorch_model-*.bin"))
    if sharded_pytorch:
        model_files.append(f"{len(sharded_pytorch)} sharded pytorch files")
    
    click.echo(f"Model files: {', '.join(model_files)}")
    
    if missing_configs:
        click.echo(f"Missing config files: {', '.join(missing_configs)}")

if __name__ == "__main__":
    cli()