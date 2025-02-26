import click
import logging

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
@click.argument("predictions_path", type=str)
def run(predictions_path: str):
    """Run analysis on processed prediction files."""
    results_df = run_analysis(predictions_path)

if __name__ == "__main__":
    cli()
