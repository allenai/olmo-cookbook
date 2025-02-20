import click

from cookbook.cli.utils import (
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


def conversion_options(func):
    @click.option("-i", "--input-dir", type=str, required=True, help="Input directory")
    @click.option(
        "-t", "--olmo-type", type=click.Choice(OLMO_TYPES), required=True, help="Type of OLMo model"
    )
    @click.option("--huggingface-tokenizer", type=str, default=None, help="Huggingface tokenizer")
    @click.option(
        "--unsharded-output-dir", type=str, default=None, help="Unsharded output directory"
    )
    @click.option(
        "--huggingface-output-dir", type=str, default=None, help="Huggingface output directory"
    )
    @click.option(
        "--unsharded-output-suffix", type=str, default="unsharded", help="Unsharded output suffix"
    )
    @click.option(
        "--huggingface-output-suffix", type=str, default="hf", help="Huggingface output suffix"
    )
    @click.option(
        "--olmoe-commit-hash", type=str, default=OLMOE_COMMIT_HASH, help="OLMoE commit hash"
    )
    @click.option(
        "--olmo2-commit-hash", type=str, default=OLMO2_COMMIT_HASH, help="OLMo2 commit hash"
    )
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
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def evaluation_options(func):
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
    @click.option(
        "-l", "--gantry-args", type=str, default="", help="Extra arguments to pass to Gantry"
    )
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
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
