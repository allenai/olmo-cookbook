import ast
import logging
from typing import List, Tuple, cast

import click
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.utils import get_default_device, seed_all
from torch.distributed.elastic.multiprocessing.errors import record

from cookbook.aliases import SourceInstance
from cookbook.model.builder import TransformerConfigBuilder

logger = logging.getLogger(__name__)


class PythonLiteralOption(click.Option):
    """
    Custom click option to parse python literals.
    """

    def type_cast_value(self, ctx, value):
        try:
            parsed = [item.replace(" ", "").replace("'", "") for item in value]
            return [ast.literal_eval(item) for item in parsed]
        except:
            raise click.BadParameter(value)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--max-tokens",
    "-t",
    type=int,
    help="Max tokens for the mixture dataset",
    required=True,
)
@click.option(
    "--source",
    "-s",
    multiple=True,
    type=str,
    help="Source datasets in the form of `Tuple[str, List[str], float]`",
    cls=PythonLiteralOption,
)
@click.option(
    "--run-name",
    "-n",
    type=str,
    help="Name of the run",
    required=True,
)
@click.option(
    "--sequence-length",
    "-l",
    type=int,
    help="Sequence length for the transformer",
)
@click.option(
    "--seed",
    "-S",
    type=int,
    help="Seed for the experiment",
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
    "--cluster",
    "-c",
    type=str,
    help="Cluster running the experiment",
)
@click.option(
    "--dtype",
    "-d",
    type=str,
    help="Data type for the dataset",
)
@click.option(
    "--tokenizer",
    "-T",
    type=str,
    help="Tokenizer for the dataset",
)
@click.option(
    "--model-identifier",
    "-m",
    type=str,
    help="Model identifier",
)
@click.option(
    "-w",
    "--weka",
    type=bool,
    default=False,
    help="Use Weka as root dir",
)
@record
def train(
    run_name: str,
    max_tokens: int,
    source: List[Tuple[str, List[str], str, str]],
    sequence_length: int,
    seed: int,
    group_id: str,
    beaker_user: str,
    cluster: str,
    dtype: str,
    tokenizer: str,
    model_identifier: str,
    weka: bool,
):
    """
    Launch a training run with the given parameters.
    """

    # Rebuild the source instances from the parsed tuples
    sources: List[SourceInstance] = []
    for item in source:
        name, paths, ratio, repetition = item
        sources.append(
            SourceInstance(
                name=name, paths=paths, ratio=float(ratio), repetition_factor=float(repetition)
            )
        )

    config = TransformerConfigBuilder(
        beaker_user=beaker_user,
        cluster=cluster,
        group_id=group_id.strip(),
        run_name=run_name.strip(),
        max_tokens=max_tokens,
        sources=sources,
        sequence_length=sequence_length,
        seed=seed,
        dtype=dtype.strip(),
        tokenizer=tokenizer.strip(),
        model_identifier=model_identifier.strip(),
        weka=weka,
    ).build()
    dataset = config.dataset.build()

    device = get_default_device()
    world_mesh = config.model.build_mesh(device=device)

    seed_all(config.init_seed)
    model = config.model.build(
        init_device="meta",
        device=device,
        mesh=world_mesh,
    )
    optim = config.optim.build(model)
    data_loader = config.data_loader.build(dataset)
    trainer = config.trainer.build(model, optim, data_loader)
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
