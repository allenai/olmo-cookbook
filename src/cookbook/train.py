import logging
from pathlib import Path

import click
from olmo_core.train import prepare_training_environment, teardown_training_environment
from torch.distributed.elastic.multiprocessing.errors import record

from cookbook.utils.config import build_train_config

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--run-name",
    "-n",
    type=str,
    help="Name of the run",
    required=True,
)
@click.option(
    "--group-id",
    "-g",
    type=str,
    help="Group ID for the experiment",
)
@click.option(
    "--beaker-user",
    "-u",
    type=str,
    help="Beaker user",
)
@click.option(
    "--config-path",
    "-C",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
@record
def train(
    run_name: str,
    group_id: str,
    beaker_user: str,
    config_path: Path,
):
    trainer = build_train_config(config_path, run_name, group_id, beaker_user)

    if trainer is None:
        logger.error("Failed to build training config! Exiting...")
        raise click.Abort()

    trainer.fit()


if __name__ == "__main__":
    try:
        prepare_training_environment()
        cli()
    finally:
        teardown_training_environment()
