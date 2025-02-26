import click
import logging
from pathlib import Path

from cookbook.analysis.process.aws import mirror_s3_to_local, process_local_folder
from cookbook.analysis.run_analysis import run_analysis
from cookbook.analysis.utils import DATA_DIR

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """CLI tools for analyzing OLMo evaluation results."""
    pass

@cli.command()
@click.option(
    "--bucket-name",
    type=str,
    default="ai2-llm",
    help="S3 bucket name containing evaluation results",
)
@click.option(
    "--s3-prefix",
    type=str,
    multiple=True,
    help="S3 prefix(es) for evaluation results. Can be specified multiple times.",
)
@click.option(
    "--local-results-path",
    type=str,
    required=True,
    help="Local path to store results",
)
@click.option(
    "--max-threads",
    type=int,
    default=100,
    help="Maximum number of threads for S3 download",
)
def download(bucket_name: str, s3_prefix: list[str], local_results_path: str, max_threads: int):
    """Download and process evaluation results from S3."""
    local_dir = f"{DATA_DIR}/{local_results_path}"
    
    # Download from S3
    mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=max_threads)
    
    # Process files to parquet
    metrics_path     = process_local_folder(local_results_path, file_type='metrics')
    predictions_path = process_local_folder(local_results_path, file_type='predictions')


@cli.command()
@click.option(
    "--local-results-path",
    type=str,
    required=True,
    help="Local path to results file",
)
def run(local_results_path: str):
    """Run analysis on processed prediction files."""
    data_dir = Path(DATA_DIR).resolve()
    prediction_path = data_dir / f"{local_results_path}_predictions.parquet"
    metrics_path    = data_dir / f"{local_results_path}_metrics.parquet"
    
    results_df = run_analysis(prediction_path)

if __name__ == "__main__":
    cli()
