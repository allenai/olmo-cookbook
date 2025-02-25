import logging
import re

import click

from cookbook.cli.utils import (
    get_aws_access_key_id,
    get_aws_secret_access_key,
    get_huggingface_token,
)
from cookbook.constants import (
    ALL_NAMED_GROUPS,
    OLMO2_COMMIT_HASH,
    OLMO_CORE_COMMIT_HASH,
    OLMO_TYPES,
    OLMOE_COMMIT_HASH,
    TRANSFORMERS_COMMIT_HASH,
)
from cookbook.eval.conversion import convert_checkpoint
from cookbook.eval.evaluation import evaluate_checkpoint

logger = logging.getLogger(__name__)


@click.argument("input_dir", type=str)
@click.option("-t", "--olmo-type", type=click.Choice(OLMO_TYPES), required=True, help="Type of OLMo model")
@click.option("--huggingface-tokenizer", type=str, default=None, help="Huggingface tokenizer")
@click.option("--unsharded-output-dir", type=str, default=None, help="Unsharded output directory")
@click.option("--huggingface-output-dir", type=str, default=None, help="Huggingface output directory")
@click.option("--unsharded-output-suffix", type=str, default="unsharded", help="Unsharded output suffix")
@click.option("--huggingface-output-suffix", type=str, default="hf", help="Huggingface output suffix")
@click.option("--olmoe-commit-hash", type=str, default=OLMOE_COMMIT_HASH, help="OLMoE commit hash")
@click.option("--olmo2-commit-hash", type=str, default=OLMO2_COMMIT_HASH, help="OLMo2 commit hash")
@click.option("--olmo-core-commit-hash", type=str, default=OLMO_CORE_COMMIT_HASH, help="OLMo core commit hash")
@click.option("--huggingface-transformers-commit-hash", type=str, default=TRANSFORMERS_COMMIT_HASH)
@click.option("--huggingface-token", type=str, default=get_huggingface_token(), help="Huggingface token")
@click.option("-b", "--use-beaker", is_flag=True, help="Use Beaker")
@click.option("--beaker-workspace", type=str, default="ai2/oe-data", help="Beaker workspace")
@click.option("--beaker-priority", type=str, default="high", help="Beaker priority")
@click.option("--beaker-cluster", type=str, default="aus", help="Beaker cluster")
@click.option("--beaker-allow-dirty", is_flag=True, help="Allow dirty Beaker workspace")
@click.option("--beaker-budget", type=str, default="ai2/oe-data", help="Beaker budget")
@click.option(
    "--beaker-preemptible/--no-beaker-preemptible", is_flag=True, help="Use preemptible instances for Beaker"
)
@click.option("--beaker-gpus", type=int, default=1, help="Number of GPUs for Beaker")
@click.option("--beaker-dry-run", is_flag=True, help="Dry run for Beaker")
@click.option("--use-system-python", is_flag=True, help="Whether to use system Python or a virtual environment")
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
    olmo_core_commit_hash: str,
    huggingface_transformers_commit_hash: str,
    unsharded_output_dir: str | None,
    unsharded_output_suffix: str,
    use_system_python: bool,
    use_beaker: bool,
    env_name: str,
    beaker_preemptible: bool,
):
    convert_checkpoint(
        beaker_allow_dirty=beaker_allow_dirty,
        beaker_budget=beaker_budget,
        beaker_cluster=beaker_cluster,
        beaker_dry_run=beaker_dry_run,
        beaker_gpus=beaker_gpus,
        beaker_preemptible=beaker_preemptible,
        beaker_priority=beaker_priority,
        beaker_workspace=beaker_workspace,
        huggingface_output_dir=huggingface_output_dir,
        huggingface_output_suffix=huggingface_output_suffix,
        huggingface_token=huggingface_token,
        huggingface_tokenizer=huggingface_tokenizer,
        huggingface_transformers_commit_hash=huggingface_transformers_commit_hash,
        input_dir=input_dir,
        olmo_core_commit_hash=olmo_core_commit_hash,
        olmo_type=olmo_type,
        olmo2_commit_hash=olmo2_commit_hash,
        olmoe_commit_hash=olmoe_commit_hash,
        python_venv_force=force_venv,
        python_venv_name=env_name,
        unsharded_output_dir=unsharded_output_dir,
        unsharded_output_suffix=unsharded_output_suffix,
        use_beaker=use_beaker,
        use_system_python=use_system_python,
    )


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
@click.option("-i", "--beaker-image", type=str, default=None, help="Beaker image to use for evaluation")
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
    default="vllm",
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
@click.option(
    "--vllm-memory-utilization",
    default=0.8,
    type=click.FloatRange(0.0, 1.0),
    help="Memory utilization for vLLM models, as a fraction of total GPU memory (between 0 and 1)",
)
@click.option(
    "--vllm-for-mc/--no-vllm-for-mc",
    default=True,
    type=bool,
    help="Whether to use hack for vLLM models for multiple choice tasks",
)
@click.option(
    "--compute-gold-bpb/--no-compute-gold-bpb",
    default=True,
    type=bool,
    help="Whether to compute gold BPB when evaluating generative tasks.",
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
    vllm_memory_utilization: float,
    vllm_for_mc: bool,
    compute_gold_bpb: bool,
):
    """Evaluate a checkpoint using the oe-eval toolkit.
    This command will launch a job on Beaker to evaluate the checkpoint using the specified parameters.
    The evaluation results will be saved to the specified remote output prefix.
    """

    # Remove any escaped hyphens in extra_args
    extra_args = re.sub(r"\\-", "-", extra_args.strip())

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
        python_venv_force=force_venv,
        python_venv_name=env_name,
        vllm_memory_utilization=vllm_memory_utilization,
        vllm_for_mc=vllm_for_mc,
        compute_gold_bpb=compute_gold_bpb,
    )


@click.group()
def cli():
    pass


cli.command()(convert)
cli.command()(evaluate)

if __name__ == "__main__":
    cli({})
