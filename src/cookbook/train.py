import logging
from pathlib import Path
from typing import cast

import click
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.utils import get_default_device, seed_all
from torch.distributed.elastic.multiprocessing.errors import record

from cookbook.model.builder import TransformerConfigBuilder
from cookbook.utils.config import config_from_path, mk_source_instances
from cookbook.utils.data import normalize_source_paths

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
    """
    Launch a training run with the given parameters.
    """

    base_config = config_from_path(config_path)

    # Because this is happening on-box in Beaker we want paths normalized for usage there.
    source_instances = mk_source_instances(normalize_source_paths(base_config.dataset.sources), None)
    dp_world_size = base_config.nodes * base_config.gpus

    config = TransformerConfigBuilder(
        beaker_user=beaker_user,
        cluster=base_config.cluster,
        downstream_evaluator=base_config.downstream_evaluator,
        dtype=base_config.dataset.dtype,
        eval_interval=base_config.eval_interval,
        group_id=group_id.strip(),
        lm_evaluator=base_config.lm_evaluator,
        max_dp_world_size=dp_world_size,
        max_target_sequence_length=base_config.max_target_sequence_length,
        max_tokens=base_config.max_tokens,
        model_identifier=base_config.model,
        run_name=run_name.strip(),
        save_interval=base_config.save_interval,
        seed=base_config.seed,
        sequence_length=base_config.sequence_length,
        sources=source_instances,
        tokenizer=base_config.tokenizer,
        wandb_config=base_config.wandb,
        weka=base_config.weka,
    ).build()

    dataset = config.dataset.build()
    device = get_default_device()
    world_mesh = config.model.build_mesh(device=device)

    seed_all(config.init_seed)
    model = config.model.build(
        init_device="meta",
        device=device,
        mesh=world_mesh,
        max_seq_len=config.dataset.sequence_length,
    )
    optim = config.optim.build(model)
    data_loader = config.data_loader.build(dataset=dataset, mesh=world_mesh)
    trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


if __name__ == "__main__":
    try:
        prepare_training_environment()
        cli()
    finally:
        teardown_training_environment()
