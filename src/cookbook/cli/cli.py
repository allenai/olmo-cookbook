import concurrent.futures
import logging
from pathlib import Path
from typing import Optional

import click
import yaml
from beaker import Beaker
from beaker.services.job import JobClient
from tqdm import tqdm
from yaspin import yaspin

from cookbook.aliases import ExperimentConfig, LaunchGroup, validate_sources
from cookbook.cli.eval import convert, evaluate
from cookbook.utils.config import build_train_config, config_from_path, mk_experiment_group, mk_launch_configs
from cookbook.utils.data import get_token_counts_and_ratios
from olmo_core.utils import generate_uuid, prepare_cli_environment

logger = logging.getLogger(__name__)


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment group configuration(s) without launching.",
)
@click.option(
    "--no-cache",
    "-n",
    is_flag=True,
    default=False,
    help="Ignore cached source details for this experiment launch.",
)
def launch(config: Path, dry_run: bool, no_cache: bool, group_id: Optional[str] = None):
    """Launch an experiment."""

    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data, path=config)
    validate_sources(experiment_config.dataset.sources)

    token_universe = get_token_counts_and_ratios(
        experiment_config.dataset.sources, experiment_config.dataset.dtype, not no_cache
    )

    if group_id:
        group_uuid = group_id
    else:
        group_uuid = generate_uuid()[:8]

    beaker_user = (Beaker.from_env().account.whoami().name).upper()  # pyright: ignore
    logger.info(f"Launching experiment group '{group_uuid}' as user '{beaker_user}'")

    logger.info(experiment_config)
    logger.info("Token distribution by source:")
    logger.info(token_universe)
    logger.info(f"Running with trainer config:")
    logger.info(build_train_config(config, experiment_config.name, group_uuid, beaker_user, dry_run=True))
    if not click.confirm("Proceed with this configuration?", default=False):
        logger.info("Launch cancelled!")
        return

    with yaspin(text="Building launch config...", color="yellow") as spinner:
        launch_group = LaunchGroup(
            instances=mk_launch_configs(
                group=mk_experiment_group(
                    config=experiment_config,
                    priors=token_universe,
                    group_uuid=group_uuid,
                ),
                beaker_user=beaker_user,
            )
        )
        spinner.ok("✔")

    with yaspin(text="Launching experiment group...", color="yellow") as spinner:
        try:
            if dry_run:
                logger.info("Dry run mode enabled. Printing experiment configurations...")
                for experiment in launch_group.instances:
                    logger.info(experiment.build_experiment_spec())
                return

            results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(experiment.launch) for experiment in launch_group.instances]

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Launching experiments",
                ):
                    results.append(future.result())

            spinner.ok("✔")
            logger.info(results)

            if results:
                logger.info(f"Experiment group '{group_uuid}' launched successfully!")
            else:
                logger.error(f"Nothing to launch for group '{group_uuid}', exiting...")
        except KeyboardInterrupt:
            logger.warning(
                "\nAborting experiment group launch! You may need to manually stop the launched experiments."
            )


def _status_for_group(path: Path, group_id: str):
    beaker = Beaker.from_env()  # pyright: ignore
    client = JobClient(beaker=beaker)  # pyright: ignore
    config = config_from_path(path)
    cluster = beaker.cluster.get(config.cluster)
    jobs = client.list(cluster=cluster)

    statuses = [
        {"status": job.status, "display_name": job.display_name}
        for job in jobs
        if job.display_name.startswith(f"{config.name}-{group_id}")
    ]
    statuses.sort(key=lambda x: x["display_name"])
    logger.info(statuses)


def _stop_for_group(path: Path, group_id: str):
    beaker = Beaker.from_env()  # pyright: ignore
    client = JobClient(beaker=beaker)  # pyright: ignore
    config = config_from_path(path)
    cluster = beaker.cluster.get(config.cluster)
    jobs = [
        {"id": job.id, "display_name": job.display_name, "status": job.status}
        for job in client.list(cluster=cluster)
        if job.display_name.startswith(f"{config.name}-{group_id}")
    ]

    if len(jobs) == 0:
        logger.info(f"No jobs found for group {group_id}")
        return

    jobs.sort(key=lambda x: x["display_name"])
    logger.info("Jobs to cancel:")
    logger.info(jobs)

    if click.confirm("Cancel these jobs?", default=False):
        for job in jobs:
            logger.info(f"Stopping job {job['display_name']}...")
            client.stop(job["id"])


@cli.command()
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to stop.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def status(config: Path, group_id: str):
    """Get the status of a launched experiment group."""

    _status_for_group(config, group_id)


@cli.command()
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to stop.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def cancel(config: Path, group_id: str):
    """Cancel all running jobs for an experiment group."""

    _stop_for_group(config, group_id)


cli.command()(evaluate)
cli.command()(convert)


if __name__ == "__main__":
    cli({})
