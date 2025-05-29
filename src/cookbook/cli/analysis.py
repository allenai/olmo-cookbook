import json
import logging

import click
import pandas as pd

from cookbook.analysis.constants import get_cache_path
from cookbook.analysis.runners import run_instance_analysis
from cookbook.eval.datalake import FindExperiments, PredictionsAll

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


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
    """Download and process evaluation results from the datalake to a dataframe."""
    cache_dir = get_cache_path(dashboard)
    prediction_path = cache_dir / f"{dashboard}_predictions.parquet"

    experiments = FindExperiments.run(dashboard=dashboard)

    logger.info(f"Found {len(experiments)} experiments in dashboard {dashboard}")

    task_predictions = PredictionsAll.prun(
        experiment_id=[experiment.experiment_id for experiment in experiments],
        force=[force for _ in experiments],
        skip_on_fail=[skip_on_fail for _ in experiments],
    )

    df = PredictionsAll.to_parquet(task_predictions)
    df.to_parquet(prediction_path)

    logger.info(f"Saved predictions to {prediction_path}")


@cli.command()
@click.option(
    "--dashboard",
    type=str,
    required=True,
    help="Dashboard name to analyze",
)
@click.option(
    "--render-plots",
    type=bool,
    default=True,
    help="Whether to generate plots",
)
def run(dashboard: str, render_plots: bool = True):
    """Run analysis on prediction dataframe."""
    cache_dir = get_cache_path(dashboard)
    prediction_path = cache_dir / f"{dashboard}_predictions.parquet"
    plot_dir = cache_dir / "plot"

    logger.info(f"Loading predictions from {prediction_path}")

    # Load the results
    df = pd.read_parquet(prediction_path)

    # Run the analysis
    outcomes = run_instance_analysis(df, render_plots=render_plots, plot_dir=plot_dir)

    logger.info(f"Saved plots to {plot_dir}")

    # Dump the results to a JSON file
    for item in outcomes:
        output_path = cache_dir / f"{dashboard}_{item[0]}.json"
        js = json.loads(item[1].to_json(orient="records"))
        with open(output_path, "w") as f:
            json.dump(js, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    cli()
