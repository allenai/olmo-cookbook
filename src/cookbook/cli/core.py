import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from cookbook.cli.utils import (
    PythonEnv,
    add_secret_to_beaker_workspace,
    get_aws_access_key_id,
    get_aws_secret_access_key,
    get_beaker_token,
    install_beaker_py,
    install_olmo_core,
    make_aws_config,
    make_aws_credentials,
)

logger = logging.getLogger(__name__)


OLMO_CORE_TRAIN_COMMIT_HASH = "2f66fd95c17c9779be9930f8fb80803293c2dc30"
NAMED_MIXES_CONFIG_DIR = Path(__file__).parent.parent / "data/mixes"
OLMO_CORE_EXAMPLES_BASE_DIR = "src/scripts/train"


def guess_model_size(model: str) -> int:
    current_size = 0
    for cand_size, cand_unit in re.findall(r"(\d+)([bBmM])", model):
        mult = 1e6 if cand_unit.lower() == "m" else 1e9
        current_size = max(current_size, int(cand_size * mult))

    if current_size == 0:
        raise ValueError(f"Could not guess model size for {model}")

    return current_size


def estimate_batch_size(
    sequence_length: int,
    total_tokens: int | None = None,
    total_steps: int | None = None,
    _factor: float = 8,
) -> int:
    """
    We estimate instant critical batch size as bs = factor * sqrt(total_steps)
    """
    if total_steps:
        critical_batch_size = _factor * total_steps ** (1 / 2)
    elif total_tokens:
        critical_batch_size = ((_factor**2) * (total_tokens / sequence_length)) ** (1 / 3)
    else:
        raise ValueError("Either total_steps or total_tokens must be provided")

    safe_batch_size = int(2 ** math.floor(math.log2(critical_batch_size)))
    max_batch_size = 2**24 // sequence_length  # 16M tokens from llama 3 405B
    return min(safe_batch_size, max_batch_size)


@dataclass(frozen=True)
class ConfigLengths:
    sequence_length: int
    rank_microbatch_size: int
    global_batch_size: int
    warmup_steps: int

    @classmethod
    def parse(cls, model_script: str, env: PythonEnv, olmo_core_dir: str) -> "ConfigLengths":
        # try running in dry_run mode
        command = [env.python, model_script, "dry_run", "temporary_run_name_name", "temporary_cluster_name"]
        process = subprocess.Popen(
            shlex.split(" ".join(command)),
            cwd=olmo_core_dir,
            env=env.path(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Error dry-running {model_script}: {stderr.decode()}")

        global_batch_size: int | None = None
        sequence_length: int | None = None
        rank_microbatch_size: int | None = None
        warmup_steps: int | None = None

        # parse sequence length
        for ln in stdout.decode().split("\n"):
            if match := re.search(r"\bsequence_length=(\d+)", ln):
                sequence_length = int(match.group(1))
            elif match := re.search(r"\brank_microbatch_size=(\d+)", ln):
                rank_microbatch_size = int(match.group(1))
            elif match := re.search(r"\bglobal_batch_size=(\d+)", ln):
                global_batch_size = int(match.group(1))
            elif match := re.search(r"\bwarmup_steps=(\d+)", ln):
                warmup_steps = int(match.group(1))

        assert sequence_length is not None, f"Could not guess sequence length for {model_script}"
        assert rank_microbatch_size is not None, f"Could not guess rank microbatch size for {model_script}"
        assert global_batch_size is not None, f"Could not guess global batch size for {model_script}"
        assert warmup_steps is not None, f"Could not guess warmup steps for {model_script}"

        return cls(
            sequence_length=sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            global_batch_size=global_batch_size,
            warmup_steps=warmup_steps,
        )


@click.option(
    "-d",
    "--data-mix",
    type=str,
    required=True,
    help=(
        "Name or path of the data mix to use to train. "
        "Can be either an absolute path to a file, or name of file in the data/mixes directory."
    ),
)
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help=(
        "Name or path of the model to train. "
        "Can be either an absolute path to a file, or name of file in the OLMo Core's src/scripts/train directory."
    ),
)
@click.option(
    "-n",
    "--duration",
    type=str,
    required=True,
    help=(
        "Duration of the training in tokens (t), steps (s), or epochs (e). "
        "Can be specified as a number followed by a unit (e.g., '10000 steps'), "
        "or using scientific notation (e.g., '1e9t'). "
    ),
)
@click.option(
    "-c",
    "--cluster",
    type=str,
    default="ai2/jupiter-cirrascale-2",
    help="Cluster(s) to use for training",
)
@click.option(
    "--force-venv",
    is_flag=True,
    help="Force creation of new virtual environment.",
    default=False,
)
@click.option(
    "--env-name",
    type=str,
    default="olmo-core-env",
    help="Name of the environment to use for training.",
)
@click.option(
    "--olmo-core-commit-hash",
    type=str,
    help="Commit hash of olmo-core to use.",
    default=OLMO_CORE_TRAIN_COMMIT_HASH,
)
@click.option(
    "-g", "--num-gpus", type=int, default=8, help="Number of GPUs to use for training. Each node has 8 GPUs."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the command to run without executing it",
    default=False,
)
@click.option(
    "--use-wandb/--no-use-wandb",
    is_flag=True,
    default=True,
    help="Enable or disable logging to wandb. If logging to wandb, must have WANDB_API_KEY set in environment.",
)
@click.option(
    "--use-comet/--no-use-comet",
    is_flag=True,
    default=False,
    help="Enable or disable logging to comet. If logging to comet, must have COMET_API_KEY set in environment.",
)
@click.option(
    "-w",
    "--workspace",
    type=str,
    required=True,
    default="ai2/oe-data",
    help="Beaker workspace to use for training.",
)
@click.option(
    "-p",
    "--priority",
    type=str,
    default="high",
    help="Priority of the job.",
)
@click.option(
    "-b",
    "--budget",
    type=str,
    default="ai2/oe-base",
    help="Budget to use for the job.",
)
@click.option(
    "--run-name",
    type=str,
    help="Name of the run.",
    default=None,
)
@click.option(
    "-i",
    "--beaker-image",
    type=str,
    default=None,
    help="Beaker image to use for training.",
)
def launch(
    data_mix: str,
    model: str,
    force_venv: bool,
    env_name: str,
    duration: str,
    olmo_core_commit_hash: str,
    num_gpus: int,
    dry_run: bool,
    use_wandb: bool,
    workspace: str,
    priority: str,
    budget: str,
    use_comet: bool,
    cluster: str,
    run_name: str | None,
    beaker_image: str | None,
):
    env = PythonEnv.create(name=env_name, force=force_venv)

    # deal with installation of libraries
    install_beaker_py(env=env)
    olmo_core_dir = install_olmo_core(commit_hash=olmo_core_commit_hash, env=env)

    config_names = {str(p.stem): p for p in (Path(olmo_core_dir) / OLMO_CORE_EXAMPLES_BASE_DIR).glob("*.py")}
    assert model in config_names or (Path(olmo_core_dir) / model).exists(), (
        f"Model {model} not found in {config_names}"
    )
    model_script_path = str(config_names.get(model, Path(model)).relative_to(olmo_core_dir))

    flags: list[str] = []

    ## FIGURE OUT DATA MIX ##
    mixes_names = {str(p.stem): p for p in NAMED_MIXES_CONFIG_DIR.glob("*.txt")}
    assert data_mix in mixes_names or Path(data_mix).exists(), f"Data mix {data_mix} not found"

    with open(mixes_names.get(data_mix, data_mix)) as f:
        data_numpys = sorted([row_ for row in f if not (row_ := row.strip()).startswith("#")])

    flags.append(f"--dataset.paths='{json.dumps(data_numpys)}'")

    ## FIGURE OUT RUN NAME ##
    if run_name is None:
        run_name = f"{model}-{os.path.basename(data_mix)}-{duration}"

    ## FIGURE OUT TRAINING DURATION ##
    # extract
    duration_match = re.match(r"^(\d+(.\d+)?(e\d+)?)([tsTS])$", duration)
    assert duration_match, f"Invalid duration format {duration}"
    duration_value = int(float(duration_match.group(1)))
    duration_unit = duration_match.group(4)

    # parse unit
    duration_map = {"t": "tokens", "s": "steps"}
    if (duration_unit_parsed := duration_map.get(duration_unit.lower())) is None:
        raise ValueError(f"Invalid duration unit {duration_unit}")
    # add to flags
    flags.append(f"--trainer.max_duration.value={duration_value}")
    flags.append(f"--trainer.max_duration.unit={duration_unit_parsed}")

    ## SET UP SECRETS ##
    env_secrets = []

    beaker_token_value = get_beaker_token(env=env)  # pyright: ignore
    assert beaker_token_value, "BEAKER_TOKEN not set"
    beaker_token_name = add_secret_to_beaker_workspace(
        secret_name="BEAKER_TOKEN",
        secret_value=beaker_token_value,
        workspace=workspace,
        env=env,  # pyright: ignore
    )
    env_secrets.append({"name": "BEAKER_TOKEN", "secret": beaker_token_name})

    if use_wandb:
        wandb_api_secret_value = os.environ.get("WANDB_API_KEY")
        assert isinstance(wandb_api_secret_value, str), "WANDB_API_KEY not set"
        wandb_api_secret_name = add_secret_to_beaker_workspace(
            secret_name="WANDB_API_KEY",
            secret_value=wandb_api_secret_value,
            workspace=workspace,
            env=env,  # pyright: ignore
        )
        env_secrets.append({"name": "WANDB_API_KEY", "secret": wandb_api_secret_name})
    flags.append(f"--trainer.callbacks.wandb.enabled={use_wandb}")

    if use_comet:
        comet_api_secret_value = os.environ.get("COMET_API_KEY")
        assert isinstance(comet_api_secret_value, str), "COMET_API_KEY not set"
        comet_api_secret_name = add_secret_to_beaker_workspace(
            secret_name="COMET_API_KEY",
            secret_value=comet_api_secret_value,
            workspace=workspace,
            env=env,  # pyright: ignore
        )
        env_secrets.append({"name": "COMET_API_KEY", "secret": comet_api_secret_name})
    flags.append(f"--trainer.callbacks.comet.enabled={use_comet}")

    if any(p.startswith("s3://") for p in data_numpys):
        aws_access_key_id = get_aws_access_key_id()
        aws_secret_access_key = get_aws_secret_access_key()
        assert aws_access_key_id and aws_secret_access_key, "AWS credentials not set"
        aws_access_key_id_name = add_secret_to_beaker_workspace(
            secret_name="AWS_ACCESS_KEY_ID",
            secret_value=aws_access_key_id,
            workspace=workspace,
            env=env,  # pyright: ignore
        )
        aws_secret_access_key_name = add_secret_to_beaker_workspace(
            secret_name="AWS_SECRET_ACCESS_KEY",
            secret_value=aws_secret_access_key,
            workspace=workspace,
            env=env,  # pyright: ignore
        )
        aws_config_name = add_secret_to_beaker_workspace(
            secret_name="AWS_CONFIG",
            secret_value=make_aws_config(profile_name="S3"),
            workspace=workspace,
            env=env,  # pyright: ignore
        )

        aws_credentials_name = add_secret_to_beaker_workspace(
            secret_name="AWS_CREDENTIALS",
            secret_value=make_aws_credentials(
                aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, profile_name="S3"
            ),
            workspace=workspace,
            env=env,  # pyright: ignore
        )

        env_secrets.append({"name": "AWS_ACCESS_KEY_ID", "secret": aws_access_key_id_name})
        env_secrets.append({"name": "AWS_SECRET_ACCESS_KEY", "secret": aws_secret_access_key_name})
        env_secrets.append({"name": "AWS_CONFIG", "secret": aws_config_name})
        env_secrets.append({"name": "AWS_CREDENTIALS", "secret": aws_credentials_name})

    flags.append(f"--launch.env_secrets='{json.dumps(env_secrets)}'")

    ## SET UP BATCH SIZE, SEQUENCE LENGTH, AND WARMUP STEPS ##

    # we run the script in dry-run mode to get the config lengths (global batch size, rank microbatch size, warmup steps, sequence length)
    config_lengths = ConfigLengths.parse(model_script=model_script_path, env=env, olmo_core_dir=olmo_core_dir)

    # we estimate the batch size based on the sequence length and the duration,
    # and override the values from the dry-run
    new_batch_size = estimate_batch_size(
        sequence_length=config_lengths.sequence_length,
        total_tokens=duration_value if duration_unit_parsed == "tokens" else None,
        total_steps=duration_value if duration_unit_parsed == "steps" else None,
    )
    new_global_batch_size = new_batch_size * config_lengths.sequence_length
    new_rank_microbatch_size = new_global_batch_size // (
        config_lengths.global_batch_size // config_lengths.rank_microbatch_size
    )
    flags.append(f"--data_loader.global_batch_size={new_global_batch_size}")
    flags.append(f"--train_module.rank_microbatch_size={new_rank_microbatch_size}")

    # we warm up for 10% or the previous warmup steps, whichever is smaller
    new_total_steps = (
        duration_value if duration_unit_parsed == "steps" else (duration_value // new_global_batch_size)
    )
    new_warmup_steps = min(config_lengths.warmup_steps, int(new_total_steps * 0.1))
    flags.append(f"--train_module.scheduler.warmup_steps={new_warmup_steps}")

    ## SET UP COMMAND ##
    num_nodes = num_gpus / 8
    assert num_nodes.is_integer(), "Number of GPUs must be a multiple of 8"

    # num nodes and cluster
    if num_nodes > 1:
        flags.append(f"--launch.num_nodes={num_nodes:.0f}")
    flags.append(f"--launch.priority={priority}")
    flags.append(f"--launch.budget={budget}")
    flags.append(f"--launch.workspace={workspace}")

    # entry point
    command = "dry_run" if dry_run else "launch"

    if beaker_image is not None:
        flags.append(f"--launch.beaker_image={beaker_image}")

    # command
    command = " ".join([env.python, model_script_path, command, run_name, cluster, *flags])
    print(f"\n\nCommand:\n{command}\nFrom:\n{olmo_core_dir}\n\n")

    if dry_run:
        return

    return subprocess.run(
        shlex.split(command),
        cwd=olmo_core_dir,
        env=env.path(),
        stdout=sys.stdout,
        stderr=sys.stderr,
        stdin=sys.stdin,
    )


@click.group()
def cli():
    pass


cli.command()(launch)

if __name__ == "__main__":
    cli({})  # pyright: ignore
