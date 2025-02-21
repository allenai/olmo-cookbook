import concurrent.futures
import logging
from pathlib import Path
from typing import Optional

import click
import yaml
from beaker import Beaker
from beaker.services.job import JobClient
from olmo_core.utils import generate_uuid, prepare_cli_environment
from tqdm import tqdm
from yaspin import yaspin

from cookbook.aliases import ExperimentConfig, LaunchGroup, validate_sources
from cookbook.cli.utils import (
    PythonEnv,
    get_aws_access_key_id,
    get_aws_secret_access_key,
    get_huggingface_token,
)
from cookbook.constants import (
    ALL_NAMED_GROUPS,
    OLMO2_COMMIT_HASH,
    OLMO_TYPES,
    OLMOE_COMMIT_HASH,
)
from cookbook.eval.checkpoints import convert_checkpoint, evaluate_checkpoint
from cookbook.utils.config import (
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
def launch(config: Path, dry_run: bool, no_cache: bool, group_id: Optional[str] = None):
    """Launch an experiment."""

    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data)
    validate_sources(experiment_config.dataset.sources)

    token_universe = get_token_counts_and_ratios(
        experiment_config.dataset.sources, experiment_config.dataset.dtype, not no_cache
    )

    logger.info("Token distribution by source:")
    logger.info(token_universe)

    if group_id:
        group_uuid = group_id
    else:
        group_uuid = generate_uuid()[:8]

    beaker_user = (Beaker.from_env().account.whoami().name).upper()
    logger.info(f"Launching experiment group '{group_uuid}' as user '{beaker_user}'")

    logger.info(experiment_config)
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
                futures = [
                    executor.submit(experiment.launch) for experiment in launch_group.instances
                ]

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
    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
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
    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
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
@click.option("-i", "--input-dir", type=str, required=True, help="Input directory")
@click.option(
    "-t", "--olmo-type", type=click.Choice(OLMO_TYPES), required=True, help="Type of OLMo model"
)
@click.option("--huggingface-tokenizer", type=str, default=None, help="Huggingface tokenizer")
@click.option("--unsharded-output-dir", type=str, default=None, help="Unsharded output directory")
@click.option(
    "--huggingface-output-dir", type=str, default=None, help="Huggingface output directory"
)
@click.option(
    "--unsharded-output-suffix", type=str, default="unsharded", help="Unsharded output suffix"
)
@click.option(
    "--huggingface-output-suffix", type=str, default="hf", help="Huggingface output suffix"
)
@click.option("--olmoe-commit-hash", type=str, default=OLMOE_COMMIT_HASH, help="OLMoE commit hash")
@click.option("--olmo2-commit-hash", type=str, default=OLMO2_COMMIT_HASH, help="OLMo2 commit hash")
@click.option(
    "--huggingface-token", type=str, default=get_huggingface_token(), help="Huggingface token"
)
@click.option("-b", "--use-beaker", is_flag=True, help="Use Beaker")
@click.option("--beaker-workspace", type=str, default="ai2/oe-data", help="Beaker workspace")
@click.option("--beaker-priority", type=str, default="high", help="Beaker priority")
@click.option("--beaker-cluster", type=str, default="aus", help="Beaker cluster")
@click.option("--beaker-allow-dirty", is_flag=True, help="Allow dirty Beaker workspace")
@click.option("--beaker-budget", type=str, default="ai2/oe-data", help="Beaker budget")
@click.option("--beaker-gpus", type=int, default=1, help="Number of GPUs for Beaker")
@click.option("--beaker-dry-run", is_flag=True, help="Dry run for Beaker")
@click.option(
    "--force-venv",
    is_flag=True,
    help="Force creation of new virtual environment",
    default=False,
)
@click.option(
    "--env-name",
    type=str,
    default="oe-conversion-venv",
    help="Name of the environment to use for conversion",
)
def convert(
    beaker_allow_dirty: bool,
    beaker_budget: str,
    beaker_cluster: str,
    beaker_dry_run: bool,
    beaker_gpus: int,
    beaker_priority: str,
    beaker_workspace: str,
    force_venv: bool,
    huggingface_output_dir: str | None,
    huggingface_output_suffix: str,
    huggingface_token: str | None,
    huggingface_tokenizer: str | None,
    input_dir: str,
    olmo2_commit_hash: str,
    olmo_type: str,
    olmoe_commit_hash: str,
    unsharded_output_dir: str | None,
    unsharded_output_suffix: str,
    use_beaker: bool,
    env_name: str,
):
    convert_checkpoint(
        input_dir=input_dir,
        olmo_type=olmo_type,
        huggingface_tokenizer=huggingface_tokenizer,
        unsharded_output_dir=unsharded_output_dir,
        huggingface_output_dir=huggingface_output_dir,
        unsharded_output_suffix=unsharded_output_suffix,
        huggingface_output_suffix=huggingface_output_suffix,
        olmoe_commit_hash=olmoe_commit_hash,
        olmo2_commit_hash=olmo2_commit_hash,
        huggingface_token=huggingface_token,
        use_beaker=use_beaker,
        beaker_workspace=beaker_workspace,
        beaker_priority=beaker_priority,
        beaker_cluster=beaker_cluster,
        beaker_allow_dirty=beaker_allow_dirty,
        beaker_budget=beaker_budget,
        beaker_gpus=beaker_gpus,
        beaker_dry_run=beaker_dry_run,
        env=PythonEnv.create(name=env_name, force=force_venv),
    )


@cli.command()
@click.argument("checkpoint_path", type=str)
@click.option("-a", "--add-bos-token", is_flag=True, help="Add BOS token")
@click.option(
    "-c",
    "--cluster",
    type=str,
    default="h100",
    help="Set cluster (aus for Austin, sea for Seattle, goog for Google, or provide specific cluster name)",
)
@click.option("-d", "--dashboard", type=str, default="generic", help="Set dashboard name")
@click.option("-b", "--budget", type=str, default="ai2/oe-data", help="Set budget")
@click.option("-w", "--workspace", type=str, default="ai2/oe-data", help="Set workspace")
@click.option(
    "-t",
    "--tasks",
    type=str,
    multiple=True,
    help=(
        "Set specific tasks or tasks groups. Can be specified multiple times. "
        f"Tasks groups are: {', '.join(ALL_NAMED_GROUPS)}"
    ),
)
@click.option(
    "-p",
    "--partition-size",
    type=int,
    default=0,
    help="How many tasks to evaluate in parallel. Set to 0 (default) to evaluate all tasks in sequence.",
)
@click.option(
    "-y",
    "--priority",
    type=click.Choice(["low", "normal", "high", "urgent"]),
    default="normal",
    help="Set priority for evaluation jobs.",
)
@click.option("-n", "--num-gpus", type=int, default=1, help="Set number of GPUs")
@click.option(
    "-x",
    "--extra-args",
    type=str,
    default="",
    help="Extra arguments to pass to oe-eval toolkit",
)
@click.option("-r", "--dry-run", is_flag=True, help="Dry run (do not launch jobs)")
@click.option(
    "-s",
    "--huggingface-secret",
    type=str,
    default=get_huggingface_token(),
    help="Beaker secret to use for Hugging Face access",
)
@click.option(
    "-j",
    "--aws-access-key-id",
    type=str,
    default=get_aws_access_key_id(),
    help="AWS access key ID to use for S3 access",
)
@click.option(
    "-k",
    "--aws-secret-access-key",
    type=str,
    default=get_aws_secret_access_key(),
    help="AWS secret access key to use for S3 access",
)
@click.option("-l", "--gantry-args", type=str, default="", help="Extra arguments to pass to Gantry")
@click.option(
    "-i", "--beaker-image", type=str, default=None, help="Beaker image to use for evaluation"
)
@click.option(
    "-z",
    "--batch-size",
    type=int,
    default=0,
    help="Set batch size for inference; if 0, use default batch size",
)
@click.option(
    "-o",
    "--remote-output-prefix",
    type=str,
    default="s3://ai2-llm/evaluation",
    help="Set remote output directory",
)
@click.option(
    "-v",
    "--model-backend",
    type=click.Choice(["hf", "vllm"]),
    default="hf",
    help="Model backend (hf for Hugging Face, vllm for vLLM)",
)
@click.option("-g", "--use-gantry", is_flag=True, help="Submit jobs with gantry directly.")
@click.option(
    "--oe-eval-commit",
    type=str,
    default=None,
    help="Commit hash of the oe-eval toolkit to use; if not provided, use the latest commit",
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
def evaluate(
    oe_eval_commit: str,
    checkpoint_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    workspace: str,
    cluster: str,
    huggingface_secret: str,
    add_bos_token: bool,
    budget: str,
    priority: str,
    num_gpus: int,
    dashboard: str,
    model_backend: str,
    tasks: list[str],
    partition_size: int,
    remote_output_prefix: str,
    extra_args: str,
    batch_size: int,
    dry_run: bool,
    beaker_image: str,
    use_gantry: bool,
    gantry_args: str,
    force_venv: bool,
    env_name: str,
):
    """Evaluate a checkpoint using the oe-eval toolkit.
    This command will launch a job on Beaker to evaluate the checkpoint using the specified parameters.
    The evaluation results will be saved to the specified remote output prefix.
    """

    evaluate_checkpoint(
        oe_eval_commit=oe_eval_commit,
        checkpoint_path=checkpoint_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        workspace=workspace,
        cluster=cluster,
        huggingface_secret=huggingface_secret,
        add_bos_token=add_bos_token,
        budget=budget,
        priority=priority,
        num_gpus=num_gpus,
        dashboard=dashboard,
        model_backend=model_backend,
        tasks=tasks,
        partition_size=partition_size,
        remote_output_prefix=remote_output_prefix,
        extra_args=extra_args,
        batch_size=batch_size,
        dry_run=dry_run,
        beaker_image=beaker_image,
        use_gantry=use_gantry,
        gantry_args=gantry_args,
        env=PythonEnv.create(name=env_name, force=force_venv),
    )


if __name__ == "__main__":
    cli({})
