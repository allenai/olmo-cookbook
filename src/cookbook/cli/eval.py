import json
import logging
import re
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from cookbook.cli.utils import (
    get_aws_access_key_id,
    get_aws_secret_access_key,
    get_huggingface_token,
    make_eval_run_name,
)
from cookbook.constants import (
    FIM_TOKENS,
    OLMO2_COMMIT_HASH,
    OLMO_CORE_COMMIT_HASH,
    OLMO_CORE_CONVERT_DTYPES,
    OLMO_CORE_V2_COMMIT_HASH,
    OLMO_TYPES,
    OLMOE_COMMIT_HASH,
    TRANSFORMERS_COMMIT_HASH,
    TRANSFORMERS_GIT_URL,
)
from cookbook.eval.named_tasks import NamedTasksGroupRegistry
from cookbook.eval.conversion import run_checkpoint_conversion
from cookbook.eval.conversion_from_hf import run_checkpoint_conversion_from_hf
from cookbook.eval.datalake import AddToDashboard, FindExperiments, RemoveFromDashboard
from cookbook.eval.evaluation import evaluate_checkpoint
from cookbook.eval.results import (
    make_dashboard_table,
    print_missing_tasks,
    find_missing_tasks,
    make_results_from_dashboard
)

logger = logging.getLogger(__name__)


@click.argument("input_dir", type=str)
@click.option("--output-dir", type=str, default=None, help="Output directory")
@click.option("--output-suffix", type=str, default="olmo_core", help="Output suffix")
@click.option(
    "--olmo-core-v2-commit-hash", type=str, default=OLMO_CORE_V2_COMMIT_HASH, help="OLMo core commit hash"
)
@click.option("--huggingface-transformers-git-url", type=str, default=TRANSFORMERS_GIT_URL)
@click.option("--huggingface-transformers-commit-hash", type=str, default=TRANSFORMERS_COMMIT_HASH)
@click.option("--huggingface-token", type=str, default=get_huggingface_token(), help="Huggingface token")
@click.option("-b", "--use-beaker", is_flag=True, help="Use Beaker")
@click.option("--beaker-workspace", type=str, default="ai2/oe-data", help="Beaker workspace")
@click.option("--beaker-priority", type=str, default="high", help="Beaker priority")
@click.option("--beaker-cluster", type=str, default="aus", help="Beaker cluster")
@click.option("--beaker-allow-dirty", is_flag=True, help="Allow dirty Beaker workspace")
@click.option("--beaker-budget", type=str, default="ai2/oe-base", help="Beaker budget")
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
    default="oe-conversion-from-hf-venv",
    help="Name of the environment to use for conversion",
)
@click.option(
    "--olmo-core-v2-experiment-json-path",
    default=None,
    help="Path to the OLMo core v2 experiment config json.",
)
@click.option(
    "--olmo-core-v2-model-arch",
    default=None,
    help=(
        "OLMo Core v2 model architecture corresponding to the HF model. "
        "New architectures should be added to ``_get_transformer_config`` in ``convert_checkpoint_from_hf.py`` of OLMo core. "
        "This is required for OLMo Core v2 when an experiment config is not provided."
    ),
)
@click.option(
    "--olmo-core-v2-tokenizer",
    default=None,
    help=(
        "OLMo Core v2 tokenizer corresponding to the HF model. "
        "New architectures should be added to ``_get_transformer_config`` in ``convert_checkpoint_from_hf.py`` of OLMo core. "
        "This is required for OLMo Core v2 when an experiment config is not provided."
    ),
)
@click.option(
    "--huggingface-transformers-model-id",
    default=None,
    help="Model id of the HF Hub repo corresponding to the model. Use to get model specific mappings in :mod:`olmo_core.nn.hf.convert`",
)
@click.option(
    "--huggingface-transformers-revision",
    default="main",
    help="Huggingface model revision/branch.",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip validation of the model after conversion.",
)
@click.option(
    "--debug-validation",
    is_flag=True,
    help="Provide extra debug logs during validation of the model after conversion.",
)
@click.option(
    "--torch-device",
    default=None,
    help="The torch device on which to run conversion and validation.",
)
def convert_checkpoint_from_hf(
    beaker_allow_dirty: bool,
    beaker_budget: str,
    beaker_cluster: str,
    beaker_dry_run: bool,
    beaker_gpus: int,
    beaker_priority: str,
    beaker_workspace: str,
    force_venv: bool,
    huggingface_token: Optional[str],
    input_dir: str,
    output_dir: Optional[str],
    output_suffix: str,
    olmo_core_v2_commit_hash: str,
    olmo_core_v2_experiment_json_path: Optional[str],
    olmo_core_v2_model_arch: Optional[str],
    olmo_core_v2_tokenizer: Optional[str],
    huggingface_transformers_git_url: str,
    huggingface_transformers_commit_hash: str,
    huggingface_transformers_model_id: Optional[str],
    huggingface_transformers_revision: str,
    use_system_python: bool,
    use_beaker: bool,
    env_name: str,
    beaker_preemptible: bool,
    skip_validation: bool,
    debug_validation: bool,
    torch_device: Optional[str],
):
    run_checkpoint_conversion_from_hf(
        beaker_allow_dirty=beaker_allow_dirty,
        beaker_budget=beaker_budget,
        beaker_cluster=beaker_cluster,
        beaker_dry_run=beaker_dry_run,
        beaker_gpus=beaker_gpus,
        beaker_preemptible=beaker_preemptible,
        beaker_priority=beaker_priority,
        beaker_workspace=beaker_workspace,
        huggingface_token=huggingface_token,
        huggingface_transformers_git_url=huggingface_transformers_git_url,
        huggingface_transformers_commit_hash=huggingface_transformers_commit_hash,
        huggingface_transformers_model_id=huggingface_transformers_model_id,
        huggingface_transformers_revision=huggingface_transformers_revision,
        input_dir=input_dir.rstrip("/"),
        output_dir=output_dir,
        output_suffix=output_suffix,
        olmo_core_v2_commit_hash=olmo_core_v2_commit_hash,
        olmo_core_v2_experiment_json_path=olmo_core_v2_experiment_json_path,
        olmo_core_v2_model_arch=olmo_core_v2_model_arch,
        olmo_core_v2_tokenizer=olmo_core_v2_tokenizer,
        python_venv_force=force_venv,
        python_venv_name=env_name,
        use_beaker=use_beaker,
        use_system_python=use_system_python,
        skip_validation=skip_validation,
        debug_validation=debug_validation,
        torch_device=torch_device,
    )


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
@click.option(
    "--olmo-core-v2-commit-hash", type=str, default=OLMO_CORE_V2_COMMIT_HASH, help="OLMo core commit hash"
)
@click.option("--huggingface-transformers-git-url", type=str, default=TRANSFORMERS_GIT_URL)
@click.option("--huggingface-transformers-commit-hash", type=str, default=TRANSFORMERS_COMMIT_HASH)
@click.option("--huggingface-token", type=str, default=get_huggingface_token(), help="Huggingface token")
@click.option("-b", "--use-beaker", is_flag=True, help="Use Beaker")
@click.option("--beaker-workspace", type=str, default="ai2/oe-data", help="Beaker workspace")
@click.option("--beaker-priority", type=str, default="high", help="Beaker priority")
@click.option("--beaker-cluster", type=str, default="aus", help="Beaker cluster")
@click.option("--beaker-allow-dirty", is_flag=True, help="Allow dirty Beaker workspace")
@click.option("--beaker-budget", type=str, default="ai2/oe-base", help="Beaker budget")
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
@click.option(
    "--max-sequence-length",
    type=int,
    default=None,
    help="Maximum sequence length of the model (olmo-core only)",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="If converting OLMo Core v2 checkpoint, skip validation of the model after conversion.",
)
@click.option(
    "--dtype",
    type=click.Choice(OLMO_CORE_CONVERT_DTYPES),
    default=None,
    help="If converting OLMo Core v2 checkpoint, the dtype to convert model weights to.",
)
def convert_checkpoint(
    beaker_allow_dirty: bool,
    beaker_budget: str,
    beaker_cluster: str,
    beaker_dry_run: bool,
    beaker_gpus: int,
    beaker_priority: str,
    beaker_workspace: str,
    force_venv: bool,
    huggingface_output_dir: Optional[str],
    huggingface_output_suffix: str,
    huggingface_token: Optional[str],
    huggingface_tokenizer: Optional[str],
    input_dir: str,
    olmo2_commit_hash: str,
    olmo_type: str,
    olmoe_commit_hash: str,
    olmo_core_commit_hash: str,
    olmo_core_v2_commit_hash: str,
    huggingface_transformers_git_url: str,
    huggingface_transformers_commit_hash: str,
    unsharded_output_dir: Optional[str],
    unsharded_output_suffix: str,
    use_system_python: bool,
    use_beaker: bool,
    env_name: str,
    beaker_preemptible: bool,
    max_sequence_length: Optional[int] = None,
    skip_validation: bool = False,
    dtype: Optional[str] = None,
):
    run_checkpoint_conversion(
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
        huggingface_transformers_git_url=huggingface_transformers_git_url,
        huggingface_transformers_commit_hash=huggingface_transformers_commit_hash,
        input_dir=input_dir.rstrip("/"),
        max_sequence_length=max_sequence_length,
        olmo_core_commit_hash=olmo_core_commit_hash,
        olmo_core_v2_commit_hash=olmo_core_v2_commit_hash,
        olmo_type=olmo_type,
        olmo2_commit_hash=olmo2_commit_hash,
        olmoe_commit_hash=olmoe_commit_hash,
        python_venv_force=force_venv,
        python_venv_name=env_name,
        unsharded_output_dir=unsharded_output_dir,
        unsharded_output_suffix=unsharded_output_suffix,
        use_beaker=use_beaker,
        use_system_python=use_system_python,
        skip_validation=skip_validation,
        dtype=dtype,
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
@click.option("-b", "--budget", type=str, default="ai2/oe-base", help="Set budget")
@click.option("-w", "--workspace", type=str, default="ai2/oe-data", help="Set workspace")
@click.option(
    "-t",
    "--tasks",
    type=str,
    multiple=True,
    help=(
        "Set specific tasks or tasks groups. Can be specified multiple times. "
        f"Tasks groups are: {', '.join(NamedTasksGroupRegistry.names())}"
    ),
)
@click.option(
    "-p",
    "--partition-size",
    type=int,
    default=0,
    help="How many tasks to evaluate per job. Set to 0 (default) to evaluate all tasks in sequence. Set to 1 for maximum parallelism.",
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
    type=click.Choice(["hf", "vllm", "olmo_core"]),
    default="vllm",
    help="Model backend (hf for Hugging Face, vllm for vLLM, olmo_core for OLMoCore)",
)
@click.option("-g", "--use-gantry", is_flag=True, help="Submit jobs with gantry directly.")
@click.option("--beaker-retries", type=int, default=0, help="Number of retries for failed evals")
@click.option(
    "--oe-eval-branch",
    type=str,
    default=None,
    help="Branch of the oe-eval toolkit to use; if not provided, use main",
)
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
@click.option(
    "--task-args",
    type=str,
    default=[],
    multiple=True,
    help="Extra arguments to pass to the task config. Can specify multiple args.",
)
@click.option(
    "--model-args",
    type=str,
    default="",
    help="Extra arguments to pass to the model",
)
@click.option(
    "--revision",
    type=str,
    default=None,
    help="Revision of model in HF hub",
)
@click.option(
    "--fim-tokens",
    type=click.Choice(list(FIM_TOKENS.keys())),
    default=None,
    help="Model-specific tokens to use for infilling tasks",
)
@click.option(
    "--vllm-use-v1-spec/--no-vllm-use-v1-spec",
    default=False,
    type=bool,
    help="Whether to use v1 spec for vLLM models",
)
@click.option(
    "--use-backend-in-run-name/--no-use-backend-in-run-name",
    default=False,
    type=bool,
    help="Whether to use the backend in the run name",
)
@click.option(
    "--backfill/--no-backfill",
    default=False,
    type=bool,
    help="When True, only launch evals that are marked as 'missing' in the workspace.",
)
@click.option(
    "--name-suffix",
    type=str,
    default="",
    help="Suffix to add to the run name",
)
@click.option(
    "--num-shots",
    type=int,
    default=None,
    help="Number of shots to use for evaluation; by default, the number of shots is part of task def in oe-eval",
)
def evaluate_model(
    oe_eval_branch: str,
    oe_eval_commit: str,
    checkpoint_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    workspace: str,
    cluster: str,
    huggingface_secret: str,
    add_bos_token: bool,
    revision: str,
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
    beaker_retries: int,
    use_gantry: bool,
    gantry_args: str,
    force_venv: bool,
    env_name: str,
    vllm_memory_utilization: float,
    vllm_for_mc: bool,
    compute_gold_bpb: bool,
    model_args: str,
    task_args: list[str],
    fim_tokens: str,
    vllm_use_v1_spec: bool,
    use_backend_in_run_name: bool,
    backfill: bool,
    name_suffix: str,
    num_shots: int | None,
):
    """Evaluate a checkpoint using the oe-eval toolkit.
    This command will launch a job on Beaker to evaluate the checkpoint using the specified parameters.
    The evaluation results will be saved to the specified remote output prefix.
    """

    if (oe_eval_branch is not None or oe_eval_commit is not None) and not use_gantry:
        raise ValueError("If oe-eval branch or commit is provided, --use-gantry should be enabled.")

    # Remove any escaped hyphens in extra_args
    extra_args = re.sub(r"\\-", "-", extra_args.strip())

    parsed_task_args: dict[str, str] = {}
    for arg in task_args:
        if not (arg := arg.strip()):
            continue
        key, value = arg.split("=")
        parsed_task_args[key] = json.loads(value)

    parsed_model_args: dict[str, str] = {}
    for arg in model_args.split(","):
        if not (arg := arg.strip()):
            continue
        key, value = arg.split("=")
        parsed_model_args[key] = value

    parsed_gantry_args: dict[str, str] = {}
    for arg in gantry_args.split(","):
        if not (arg := arg.strip()):
            continue
        key, value = arg.split("=", 1)
        parsed_gantry_args[key] = value

    if backfill:
        # Get the model's run name using cookbook logic
        # E.g., "allenai/OLMo-2-1124-13B-SFT" => "OLMo-2-1124-13B-SFT"
        model_name = make_eval_run_name(
            checkpoint_path=checkpoint_path,
            add_bos_token=add_bos_token,
            num_shots=num_shots,
            name_suffix=name_suffix.strip(),
        )

        # to find what to backfill with, we get all results for this dashboard, then filter them
        # to match this model name, and finally find all missing tasks in results.
        dashboard_table = make_dashboard_table(dashboard=dashboard)
        results = make_results_from_dashboard(dashboard_table=dashboard_table, tasks=tasks, models=[model_name])
        missing_tasks = find_missing_tasks(results=results)

        # Override our tasks with the missing set
        if missing_tasks and model_name in missing_tasks:
            tasks = missing_tasks[model_name]
        else:
            print(f"Found no missing tasks for {model_name}")
            return

    evaluate_checkpoint(
        oe_eval_branch=oe_eval_branch,
        oe_eval_commit=oe_eval_commit,
        checkpoint_path=checkpoint_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        workspace=workspace,
        cluster=cluster,
        huggingface_secret=huggingface_secret,
        add_bos_token=add_bos_token,
        revision=revision,
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
        beaker_retries=beaker_retries,
        use_gantry=use_gantry,
        gantry_args=parsed_gantry_args,
        python_venv_force=force_venv,
        python_venv_name=env_name,
        vllm_memory_utilization=vllm_memory_utilization,
        vllm_for_mc=vllm_for_mc,
        compute_gold_bpb=compute_gold_bpb,
        task_args=parsed_task_args,
        model_args=parsed_model_args,
        fim_tokens=fim_tokens,
        use_vllm_v1_spec=vllm_use_v1_spec,
        use_backend_in_run_name=use_backend_in_run_name,
        name_suffix=name_suffix,
        num_shots=num_shots,
    )


@click.option("-d", "--dashboard", type=str, required=True, help="Set dashboard name")
@click.option(
    "-m",
    "--models",
    type=str,
    multiple=True,
    default=None,
    help="Set specific models to show. Can be specified multiple times.",
)
@click.option(
    "-t",
    "--tasks",
    type=str,
    required=True,
    multiple=True,
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["json", "table", "csv"]),
    default="table",
    help="Output results in JSON format",
)
@click.option(
    "-s",
    "--sort-by",
    type=click.Choice(["column", "name", "col", "average", "avg"]),
    default="column",
    help="Column sort approach (allowed values: column, name, col, average, avg)",
)
@click.option(
    "-S",
    "--sort-column-name",
    type=str,
    default="",
    help="Name of the column to sort by",
)
@click.option("-A/--ascending", "sort_descending", flag_value=False, default=False, help="Sort ascending")
@click.option("-D/--descending", "sort_descending", flag_value=True, default=False, help="Sort descending")
@click.option(
    "-F",
    "--force",
    is_flag=True,
    help="Force re-fetch results from the datalake",
)
@click.option(
    "--skip-on-fail",
    is_flag=True,
    help="Skip experiments that fail to fetch results from the datalake",
)
def get_results(
    dashboard: str,
    models: list[str],
    tasks: list[str],
    format: str,
    sort_by: str,
    sort_column_name: str,
    sort_descending: bool,
    force: bool,
    skip_on_fail: bool,
) -> None:
    # we get the metrics table from the datalake
    dashboard_table = make_dashboard_table(
        dashboard=dashboard,
        force=force,
        skip_on_fail=skip_on_fail,
    )

    # we subselect the right tasks and models, plus expand named tasks
    results = make_results_from_dashboard(
        dashboard_table=dashboard_table,
        tasks=tasks,
        models=models,
    )

    # we find missing tasks in the results
    missing_tasks = find_missing_tasks(results=results)

    # we gotta let the user know if there are any missing tasks
    print_missing_tasks(missing_tasks=missing_tasks, models=models, tasks=tasks)

    # okay we got all results! now time to sort them depending on the user's request
    try:
        results = results.sort(
            by_col=((sort_column_name or next(iter(results.columns))) if sort_by.startswith("col") else None),
            by_name=sort_by.startswith("name"),
            by_avg="avg" in sort_by or "average" in sort_by,
            reverse=not sort_descending,
        )
    except StopIteration:
        # if no columns are left, we don't need to sort
        pass

    # output according to format requested by the user
    if format == "json":
        print(results.to_json())
    elif format == "table":
        results.show()
    elif format == "csv":
        print(results.to_csv())
    else:
        raise ValueError(f"Invalid format: {format}")

    return results


@click.option("-d", "--dashboard", type=str, required=True, help="Set dashboard name")
@click.option(
    "-m",
    "--models",
    type=str,
    multiple=True,
    required=True,
    help="Models to add to the dashboard",
)
@click.option(
    "-s",
    "--source-dashboard",
    type=str,
    help="Optional argument: if provided, only copy results from this dashboard",
    default=None,
)
def add_to_dashboard(dashboard: str, models: list[str], source_dashboard: str | None) -> None:
    resp = AddToDashboard.prun(
        dashboard=[dashboard for _ in models],
        model_name=list(models),
        source_dashboard=[source_dashboard for _ in models],
    )
    print(f"Added {len(resp)} models to the dashboard")


@click.option("-d", "--dashboard", type=str, required=True, help="Set dashboard name")
@click.option(
    "-m",
    "--models",
    type=str,
    required=True,
    multiple=True,
)
def remove_from_dashboard(dashboard: str, models: list[str]) -> None:
    resp = RemoveFromDashboard.prun(dashboard=[dashboard for _ in models], model_name=list(models))
    print(f"Removed {len(resp)} models from the dashboard")


@click.option("-t", "--task", type=str, multiple=True, help="Tasks to filter by")
def list_tasks(task: list[str] | None):
    valid_tasks = [re.compile(t) if re.escape(t) != t else t for t in task] if task else []

    # first, we write a helper function to compare tasks; this is necessary because both valid tasks and
    # tasks in a group can be either strings or regex patterns.
    def _compare_tasks(task_target: str | re.Pattern, task_source: str | re.Pattern) -> bool:
        if isinstance(task_source, re.Pattern):
            if isinstance(task_target, re.Pattern):
                return task_source == task_target
            else:
                return task_source.search(task_target) is not None
        elif isinstance(task_target, re.Pattern):
            return task_target.search(task_source) is not None
        else:
            return task_target == task_source

    table = Table(title="Listing named tasks")
    table.add_column("Group")
    table.add_column("Tasks")
    table.add_column("Count")

    for group_name in NamedTasksGroupRegistry.names():
        task_group = NamedTasksGroupRegistry.get(group_name)

        # there are three option on what to print here:
        # 1. some of the tasks in the group are in valid tasks: we print the name of the group, and those tasks
        # 2. the group_name name is in valid tasks: we print the name of the group, and all tasks in the group
        # 3. none of the tasks in the group are in valid tasks: we skip the group
        # 4. the valid tasks is empty: we print the name of the group, and all tasks in the group

        # first shortcut; if valid tasks is empty, we print all tasks in the group
        if len(valid_tasks) == 0:
            table.add_row(
                group_name,
                "\n".join(str(t) for t in task_group.expanded_tasks),
                f"{len(task_group.expanded_tasks):,}",
            )
            continue

        # another shortcut; if any of valid names matches group name, we print all tasks in the group
        if any(v.search(group_name) if isinstance(v, re.Pattern) else v == group_name for v in valid_tasks):
            table.add_row(
                group_name,
                "\n".join(str(t) for t in task_group.expanded_tasks),
                f"{len(task_group.expanded_tasks):,}",
            )
            continue

        # okay, if we are here, means that the group name is not in valid tasks; but there might be some tasks
        # that are, so we need to check that.

        matching_tasks_from_group: list[str | re.Pattern] = []
        for task_item in task_group.expanded_tasks:
            for valid_task in valid_tasks:
                if _compare_tasks(task_item, valid_task):
                    matching_tasks_from_group.append(task_item)

        if len(matching_tasks_from_group) > 0:
            table.add_row(
                group_name,
                "\n".join(str(t) for t in matching_tasks_from_group),
                f"{len(matching_tasks_from_group):,}/{len(task_group.expanded_tasks):,}",
            )

    console = Console()
    console.print(table)


@click.option("-m", "--model", type=str, required=True, help="List models for a given suite")
@click.option("-t", "--task", type=str, multiple=True, help="List experiments for a given task")
def list_all_experiments(model: str, task: list[str] | None) -> None:
    experiments = FindExperiments.run(model_name=model)
    valid_tasks = [re.compile(t) for t in task] if task else []

    table = Table(title=f"Listing experiments for model {model}")
    table.add_column("Experiment ID")
    table.add_column("Model Name")
    table.add_column("Task Name")
    table.add_column("Tags")
    table.add_column("Count")
    for experiment in experiments:
        tasks_in_experiment = experiment.task_name.split(",")
        if valid_tasks:
            tasks_in_experiment = [t for t in tasks_in_experiment if any(v.search(t) for v in valid_tasks)]

        if len(tasks_in_experiment) == 0:
            continue

        table.add_row(
            experiment.experiment_id,
            experiment.model_name,
            "\n".join(tasks_in_experiment),
            "\n".join(map(str, experiment.tags)),
            f"{len(tasks_in_experiment):,}",
        )

    console = Console()
    console.print(table)


@click.group()
def cli():
    pass


cli.command("convert-from-hf")(convert_checkpoint_from_hf)
cli.command("convert")(convert_checkpoint)
cli.command("evaluate")(evaluate_model)
cli.command("results")(get_results)
cli.command("list")(list_tasks)
cli.command("experiments")(list_all_experiments)
cli.command("add-dashboard")(add_to_dashboard)
cli.command("remove-dashboard")(remove_from_dashboard)


if __name__ == "__main__":
    cli({})
