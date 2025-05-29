import json
import logging
from pathlib import Path

import click
import pandas as pd

from cookbook.analysis.constants import DATA_DIR
from cookbook.analysis.runners import run_instance_analysis
from cookbook.eval.datalake import FindExperiments, PredictionsAll

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """CLI tools for analyzing OLMo evaluation results."""
    pass


@cli.command()
@click.option(
    "--dashboard",
    type=str,
    required=True,
    help="Dashboard name to analyze",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force recomputation of metrics",
)
@click.option(
    "--skip-on-fail",
    is_flag=True,
    default=False,
    help="Skip tasks that fail instead of raising an error",
)
def download(dashboard: str, force: bool = False, skip_on_fail: bool = False):
    """Download and process evaluation results from S3."""
    data_dir = Path(DATA_DIR).resolve()

    experiments = FindExperiments.run(dashboard=dashboard)

    logger.info(f"Found {len(experiments)} experiments in dashboard {dashboard}")

    task_predictions = PredictionsAll.prun(
        experiment_id=[experiment.experiment_id for experiment in experiments],
        force=[force for _ in experiments],
        skip_on_fail=[skip_on_fail for _ in experiments],
    )

    df = PredictionsAll.to_parquet(task_predictions)

    # Save predictions to parquet file
    prediction_path = data_dir / f"{dashboard}_predictions.parquet"
    df.to_parquet(prediction_path)


@cli.command()
@click.option(
    "--dashboard",
    type=str,
    required=True,
    help="Dashboard name to analyze",
)
def run(dashboard: str):
    """Run analysis on processed prediction files."""
    data_dir = Path(DATA_DIR).resolve()
    prediction_path = data_dir / f"{dashboard}_predictions.parquet"

    # Load the results
    df = pd.read_parquet(prediction_path)

    # Run the analysis
    outcomes = run_instance_analysis(df)

    # Dump the results to a JSON file
    for item in outcomes:
        output_path = data_dir / f"{dashboard}_{item[0]}.json"
        js = json.loads(item[1].to_json(orient="records"))
        with open(output_path, "w") as f:
            json.dump(js, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    cli()
