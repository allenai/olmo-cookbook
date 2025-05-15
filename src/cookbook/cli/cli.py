import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Optional

import click
import yaml
from beaker import Beaker
from beaker.exceptions import SecretNotFound
from beaker.services.job import JobClient
from olmo_core.utils import generate_uuid, prepare_cli_environment
from tqdm import tqdm
from yaspin import yaspin

from cookbook.aliases import ExperimentConfig, LaunchGroup, validate_sources
from cookbook.cli.core import estimate_batch_size
from cookbook.cli.eval import convert_checkpoint, evaluate_model
from cookbook.utils.config import (
    build_train_config,
    config_from_path,
    mk_experiment_group,
    mk_launch_configs,
)
from cookbook.utils.data import get_token_counts_and_ratios

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
@click.option(
    "--group-id",
    "-g",
    default=None,
    help="Overrides the generated run group_id, allows for restarts with config changes or similar",
)
@click.option(
    "--visualize-schedule",
    is_flag=True,
    default=False,
    help="Beta: Visualize the learning rate schedule for the experiment",
)
def launch(
    config: Path, dry_run: bool, no_cache: bool, group_id: Optional[str] = None, visualize_schedule: bool = False
):
    """Launch an experiment."""

    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data, path=config, visualize_schedule=visualize_schedule)
    validate_sources(experiment_config.dataset.sources)

    token_universe = get_token_counts_and_ratios(
        experiment_config.dataset.sources, experiment_config.dataset.dtype, not no_cache
    )

    sequence_length = experiment_config.sequence_length
    max_tokens = experiment_config.max_tokens

    suggested_batch_size_tokens = (
        estimate_batch_size(sequence_length=sequence_length, total_tokens=max_tokens)
        * experiment_config.sequence_length
    )

    if experiment_config.global_batch_size:
        if suggested_batch_size_tokens != experiment_config.global_batch_size:
            logger.warning(
                f"Suggested global batch size {suggested_batch_size_tokens:,} is different from the configured global batch size {experiment_config.global_batch_size:,}. "
                "This may lead to suboptimal performance. Consider adjusting the batch size."
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
                    group_id=group_uuid,
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


@cli.command()
@click.option(
    "--source",
    "-s",
    type=str,
    help="The source workspace to copy secrets from",
    required=True,
)
@click.option(
    "--dest",
    "-d",
    type=str,
    help="The destination workspace to copy secrets to",
    default="ai2/dolma2",
)
def prepare_user_workspace_from(source: str, dest: str):
    beaker = Beaker.from_env()

    if source == dest:
        raise ValueError("Dest workspace cannot be source workspace")

    if not source.startswith("ai2/"):
        raise ValueError("source workspace must be in the ai2 organization")

    user = beaker.account.whoami().name.upper()
    source_workspace = beaker.workspace.get(source)
    target_workspace = beaker.workspace.get(dest)

    required = (
        f"{user}_BEAKER_TOKEN",
        f"{user}_WANDB_API_KEY",
        f"{user}_AWS_CONFIG",
        f"{user}_AWS_CREDENTIALS",
        "R2_ENDPOINT_URL",
        "WEKA_ENDPOINT_URL",
    )

    for secret_name in required:
        secret_value = beaker.secret.read(secret_name, workspace=source_workspace)
        beaker.secret.write(secret_name, secret_value, workspace=target_workspace)

        print(f"copied '{secret_name}' to {target_workspace.full_name}")


@cli.command()
@click.option(
    "--workspace",
    "-w",
    type=str,
    help="The Beaker workspace to write secrets to",
    default=None,
    required=True,
)
@click.option(
    "--beaker-token",
    "-t",
    type=str,
    help="The beaker token to use for authentication",
    default=None,
    required=True,
)
@click.option(
    "--aws-config",
    "-c",
    type=str,
    help="The AWS config file to use for authentication",
    default=None,
    required=True,
)
@click.option(
    "--aws-credentials",
    "-a",
    type=str,
    help="The AWS credentials file to use for authentication",
    default=None,
    required=True,
)
@click.option(
    "--r2-endpoint-url",
    type=str,
    help="The R2 endpoint URL to use for authentication",
    default=None,
    required=False,
)
@click.option(
    "--weka-endpoint-url",
    type=str,
    help="The WEKA endpoint URL to use for authentication",
    default=None,
    required=False,
)
@click.option(
    "--wandb-api-key",
    type=str,
    help="The wandb API key to use for authentication",
    default=None,
    required=False,
)
def prepare_user_workspace(
    workspace: str,
    beaker_token: str,
    aws_config: str,
    aws_credentials: str,
    r2_endpoint_url: Optional[str] = None,
    weka_endpoint_url: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
):
    """Prepare the workspace environment for use with OLMo-cookbook."""

    beaker = Beaker.from_env()
    user = beaker.account.whoami().name.upper()
    target_workspace = beaker.workspace.get(workspace)

    def read_file(file_path) -> str:
        """Read a file and return its contents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r") as file:
            return file.read()

    aws_config = read_file(aws_config)
    aws_credentials = read_file(aws_credentials)

    secrets: dict[str, Optional[str]] = {
        f"{user}_BEAKER_TOKEN": beaker_token,
        f"{user}_WANDB_API_KEY": wandb_api_key,
        f"{user}_AWS_CONFIG": aws_config,
        f"{user}_AWS_CREDENTIALS": aws_credentials,
        "R2_ENDPOINT_URL": r2_endpoint_url,
        "WEKA_ENDPOINT_URL": weka_endpoint_url,
    }

    for secret_name, secret_value in secrets.items():
        if secret_value:
            beaker.secret.write(secret_name, secret_value, workspace=target_workspace)
            print(f"Succesfully wrote '{secret_name}' to {target_workspace.full_name}")

        # If a workspace secret doesn't exist at this point, then write in a blank value
        try:
            beaker.secret.get(secret_name, workspace=target_workspace)
        except SecretNotFound:
            if user not in secret_name:
                beaker.secret.write(secret_name, "[blank]", workspace=target_workspace)
                print(f"Writing blank value for {secret_name}")


cli.command()(evaluate_model)
cli.command()(convert_checkpoint)


if __name__ == "__main__":
    cli({})
