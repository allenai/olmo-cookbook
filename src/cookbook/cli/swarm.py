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

from cookbook.aliases import SwarmConfig, LaunchGroup, validate_sources
from cookbook.cli.core import estimate_batch_size
from cookbook.utils.config import (

    mk_swarm_experiment_group,
    mk_launch_configs,
)
from cookbook.utils.data import get_token_counts_and_ratios
from cookbook.utils.swarm_utils import mk_mixes 

logger = logging.getLogger(__name__)


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option(
    "-c",
    "--swarm-config",
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
def swarm(swarm_config: Path, dry_run: bool, no_cache: bool, group_id: Optional[str] = None):
    """Launch an experiment."""

    with open(swarm_config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = SwarmConfig(**data, path=swarm_config)
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

    logger.info("Generating experiment group from the following config...")
    logger.info(experiment_config)
    if not click.confirm("Proceed with this configuration?", default=False):
        logger.info("Launch cancelled!")
        return
    


    mixes = mk_mixes(experiment_config, use_cache=(no_cache == False))
    if click.confirm("Launch experiment with this set of mixtures?", default=False):
        with yaspin(text="Building experiment group...", color="yellow") as spinner:
            launch_group = LaunchGroup(
                instances=mk_launch_configs(
                    group=mk_swarm_experiment_group(
                        config=experiment_config,
                        mixes=mixes,
                        group_id=group_uuid,
                    ),
                    beaker_user=beaker_user,
                )
            )


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



if __name__ == "__main__":
    cli({})
