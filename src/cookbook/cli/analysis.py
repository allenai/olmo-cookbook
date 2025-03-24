import json
import logging
from pathlib import Path

import click

from cookbook.analysis.constants import DATA_DIR
from cookbook.analysis.data.aws import mirror_s3_to_local, process_local_folder
from cookbook.analysis.runners import run_instance_analysis

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

    # Convert to parquet
    process_local_folder(local_results_path, file_type="predictions")


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

    # Run the analysis
    outcomes = run_instance_analysis(prediction_path)

    # Dump the results to a JSON file
    for item in outcomes:
        output_path = data_dir / f"{local_results_path}_{item[0]}.json"
        js = json.loads(item[1].to_json(orient="records"))
        with open(output_path, "w") as f:
            json.dump(js, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    cli()
