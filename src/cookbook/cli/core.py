import logging
from pathlib import Path
import re

import click

from cookbook.cli.utils import (
    get_aws_access_key_id,
    get_aws_secret_access_key,
)

import json
import os
import shlex
import shutil
import subprocess

import yaml

from cookbook.cli.utils import (
    PythonEnv,
    add_secret_to_beaker_workspace,
    install_olmo_core,
    get_beaker_token,
    make_aws_config,
    make_aws_credentials,
)


logger = logging.getLogger(__name__)


OLMO_CORE_TRAIN_COMMIT_HASH = "78be552957a19ba4ea6dd7ba0398bf9a0028ea65"
NAMED_MIXES_CONFIG_DIR = Path(__file__).parent.parent / "data/mixes"
OLMO_CORE_EXAMPLES_BASE_DIR = "src/scripts/train"


@click.option(
    '-d', '--data-mix',
    type=str,
    required=True,
    help="Name or path of the data mix to use to train"
)
@click.option(
    '-m', '--model',
    type=str,
    required=True,
    help="Name or path of the model to train"
)
@click.option(
    '-n', '--duration',
    type=str,
    required=True,
    help="Duration of the training"
)
@click.option(
    "-c", "--cluster",
    type=str,
    default="ai2/jupiter-cirrascale-2",
    help="Cluster(s) to use for training",
)
@click.option(
    "--force-venv",
    is_flag=True,
    help="Force creation of new virtual environment",
    default=False,
)
@click.option(
    "--env-name",
    type=str,
    default="oe-eval-venv",
    help="Name of the environment to use for evaluation",
)
@click.option(
    "--olmo-core-commit-hash",
    type=str,
    help="Commit hash of olmo-core to use",
    default=OLMO_CORE_TRAIN_COMMIT_HASH,
)
@click.option(
    "--num-nodes",
    type=int,
    default=1,
    help="Number of nodes to use for training"
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
    help="Enable or disable logging to wandb",
)
@click.option(
    "--use-comet/--no-use-comet",
    is_flag=True,
    default=False,
    help="Enable or disable logging to comet",
)
@click.option(
    "-w", "--workspace",
    type=str,
    required=True,
    default="ai2/oe-data",
    help="Beaker workspace to use for training",
)
@click.option(
    "-p", "--priority",
    type=str,
    default="high",
    help="Priority of the job",
)
@click.option(
    "-b", "--budget",
    type=str,
    default="ai2/oe-data",
    help="Budget to use for the job",
)
@click.option(
    "--run-name",
    type=str,
    help="Name of the run",
    default=None,
)
@click.option(
    "-i", "--beaker-image",
    type=str,
    default=None,
    help="Beaker image to use for training",
)
def launch(
    data_mix: str,
    model: str,
    force_venv: bool,
    env_name: str,
    duration: str,
    olmo_core_commit_hash: str,
    num_nodes: int,
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
    olmo_core_dir = install_olmo_core(commit_hash=olmo_core_commit_hash, env=env)

    config_names = {str(p.stem): p for p in (Path(olmo_core_dir) / OLMO_CORE_EXAMPLES_BASE_DIR).glob("*.py")}
    assert model in config_names or (Path(olmo_core_dir) / model).exists(), \
        f"Model {model} not found in {config_names}"

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
    duration_match = re.match(r"^(\d+(.\d+)?(e\d+)?)([tesTES])$", duration)
    assert duration_match, f"Invalid duration format {duration}"
    duration_value = int(float(duration_match.group(1)))
    duration_unit = duration_match.group(4)

    # parse unit
    duration_map = {"t": "tokens", "e": "epochs", "s": "steps"}
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
        env=env     # pyright: ignore
    )
    env_secrets.append({"name": "BEAKER_TOKEN", "secret": beaker_token_name})

    if use_wandb:
        wandb_api_secret_value = os.environ.get("WANDB_API_KEY")
        assert isinstance(wandb_api_secret_value, str), "WANDB_API_KEY not set"
        wandb_api_secret_name = add_secret_to_beaker_workspace(
            secret_name="WANDB_API_KEY",
            secret_value=wandb_api_secret_value,
            workspace=workspace,
            env=env     # pyright: ignore
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
            env=env     # pyright: ignore
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
            env=env     # pyright: ignore
        )
        aws_secret_access_key_name = add_secret_to_beaker_workspace(
            secret_name="AWS_SECRET_ACCESS_KEY",
            secret_value=aws_secret_access_key,
            workspace=workspace,
            env=env     # pyright: ignore
        )
        aws_config_name = add_secret_to_beaker_workspace(
            secret_name="AWS_CONFIG",
            secret_value=make_aws_config(),
            workspace=workspace,
            env=env     # pyright: ignore
        )

        aws_credentials_name = add_secret_to_beaker_workspace(
            secret_name="AWS_CREDENTIALS",
            secret_value=make_aws_credentials(aws_access_key_id, aws_secret_access_key),
            workspace=workspace,
            env=env     # pyright: ignore
        )

        env_secrets.append({"name": "AWS_ACCESS_KEY_ID", "secret": aws_access_key_id_name})
        env_secrets.append({"name": "AWS_SECRET_ACCESS_KEY", "secret": aws_secret_access_key_name})
        env_secrets.append({"name": "AWS_CONFIG", "secret": aws_config_name})
        env_secrets.append({"name": "AWS_CREDENTIALS", "secret": aws_credentials_name})

    flags.append(f"--launch.env_secrets='{json.dumps(env_secrets)}'")

    ## SET UP COMMAND ##

    # num nodes and cluster
    if num_nodes > 1:
        flags.append(f"--launch.num_nodes={num_nodes}")
    flags.append(f"--launch.priority={priority}")
    flags.append(f"--launch.budget={budget}")
    flags.append(f"--launch.workspace={workspace}")

    # entry point
    command = "dry_run" if dry_run else "launch"

    if beaker_image is not None:
        flags.append(f"--launch.beaker_image={beaker_image}")

    # command
    command = " ".join(
        [
            env.python,
            str(config_names.get(model, Path(model)).relative_to(olmo_core_dir)),
            command,
            run_name,
            cluster,
            *flags
        ]
    )
    print(f"\n\nCommand:\n{command}\nFrom:\n{olmo_core_dir}\n\n")
    # Run the command and stream output in real-time
    process = subprocess.Popen(
        shlex.split(command),
        cwd=olmo_core_dir,
        env=env.path(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Print output as it comes
    stdout_lines = []
    stderr_lines = []

    while True:
        stdout_line = process.stdout.readline() if process.stdout else ""
        stderr_line = process.stderr.readline() if process.stderr else ""

        if not stdout_line and not stderr_line and process.poll() is not None:
            break

        if stdout_line:
            print(stdout_line, end="")
            stdout_lines.append(stdout_line)

        if stderr_line:
            print(stderr_line, end="")
            stderr_lines.append(stderr_line)

    # Get return code
    returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(f"Error running command: {command}")


@click.group()
def cli():
    pass


cli.command()(launch)

if __name__ == "__main__":
    cli({})  # pyright: ignore
