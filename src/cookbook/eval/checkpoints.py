import re
import shlex
import subprocess
from hashlib import md5
from pathlib import Path
from urllib.parse import urlparse

from cookbook.cli.utils import (
    PythonEnv,
    add_aws_flags,
    add_secret_to_beaker_workspace,
    check_beaker_dependencies,
    discover_weka_mount,
    find_repository_root,
    install_oe_eval,
    make_eval_run_name,
    remove_conflicting_packages,
)
from cookbook.constants import (
    ALL_NAMED_GROUPS,
    BEAKER_KNOWN_CLUSTERS,
    DEFAULT_OLMO2_TOKENIZER,
    DEFAULT_OLMO_CORE_TOKENIZER,
    DEFAULT_OLMOE_TOKENIZER,
    OE_EVAL_LAUNCH_COMMAND,
    OLMO_TYPES,
    WEKA_MOUNTS,
)
from cookbook.eval.conversion import convert_olmo_checkpoint


def evaluate_checkpoint(
    oe_eval_commit: str,
    checkpoint_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    workspace: str,
    cluster: str,
    huggingface_secret: str,
    add_bos_token: bool,
    compute_gold_bpb: bool,
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
    gpu_memory_utilization: float,
    env: PythonEnv,
):
    # Install oe-eval toolkit
    oe_eval_dir = install_oe_eval(env=env, commit_hash=oe_eval_commit)

    # this is where we store all fixed flags to pass to oe-eval
    flags: list[str] = []

    # clusters_to_exclude
    clusters_to_exclude: set[str] = set()

    # Need to figure out how checkpoint is stored!
    if (scheme := urlparse(checkpoint_path).scheme) == "s3":
        print(
            "Checkpoint is stored in S3; I will add AWS credentials to workspace and pass them to oe-eval"
        )
        is_aws_cred_added = add_aws_flags(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            workspace=workspace,
            flags=flags,
        )

        if not is_aws_cred_added:
            raise ValueError("AWS access key ID and secret access key must be provided")

    elif scheme == "gs" or scheme == "gcs":
        # This is google cloud storage; for now, we just override cluster
        # so that it runs on augusta; credentials to fetch there are automatically
        # configured
        if cluster != "goog" and cluster not in BEAKER_KNOWN_CLUSTERS["goog"]:
            print(
                "Checkpoint is stored in Google Cloud Storage, "
                "but cluster is not set to 'goog'. Overriding cluster."
            )
            cluster = "goog"
    elif scheme:
        raise ValueError(f"Unsupported scheme '{scheme}' in checkpoint path")
    elif checkpoint_path.startswith("/") and any(
        re.match(rf"/{w}/", checkpoint_path) for w in WEKA_MOUNTS
    ):
        print("Checkpoint is stored in Weka; I will remove cluster that have no WEKA.")
        for cl in BEAKER_KNOWN_CLUSTERS["goog"]:
            clusters_to_exclude.add(cl)
        checkpoint_path = f"weka://{checkpoint_path.lstrip('/').rstrip('/')}"

    else:
        print("Path is a huggingface model; I will add huggingface token to workspace")
        assert huggingface_secret, "Hugging Face token must be provided"
        hf_token_secret = add_secret_to_beaker_workspace(
            secret_name="HUGGING_FACE_HUB_TOKEN",
            secret_value=huggingface_secret,
            workspace=workspace,
        )
        flags.append(f"--gantry-secret-hf-read-only '{hf_token_secret}'")
        flags.append("--gantry-args '{\"hf_token\":true}'")

    # we use this run name when storing store all the output files
    run_name = make_eval_run_name(checkpoint_path=checkpoint_path, add_bos_token=add_bos_token)

    # replace nicknames for actual cluster names, joined with commas
    beaker_clusters = ",".join(
        set(BEAKER_KNOWN_CLUSTERS.get(cluster, [cluster])).difference(clusters_to_exclude)
    )

    # set the beaker flags
    flags.append(f"--beaker-workspace '{workspace}'")
    flags.append(f"--beaker-budget '{budget}'")
    flags.append(f"--beaker-priority {priority}")
    flags.append(f"--cluster '{beaker_clusters}'")

    # set resources
    flags.append(f"--gpus {num_gpus}")

    # datalake parameters (mostly have to push there + tags)
    flags.append(f"--datalake-tags 'dashboard={dashboard},checkpoint={run_name}'")
    flags.append("--push-datalake")

    # set model info
    gpu_memory_utilization = f',gpu_memory_utilization={gpu_memory_utilization}' if gpu_memory_utilization else ''
    flags.append(f"--model {run_name}")
    flags.append(f"--model-args 'model_path={checkpoint_path},add_bos_token={add_bos_token}{gpu_memory_utilization}'")
    flags.append(f"--model-type {model_backend}")

    all_tasks = sorted(
        set(task for task_group in tasks for task in ALL_NAMED_GROUPS.get(task_group, [task_group]))
    )

    cnt = 0
    for i in range(0, len(all_tasks), partition_size or len(all_tasks)):
        # add all tasks in the partition as flag
        partition_tasks = all_tasks[i : i + partition_size] if partition_size else all_tasks
        flags.append(f"--task {' '.join(partition_tasks)}")

        if add_aws_flags(aws_access_key_id, aws_secret_access_key, workspace, flags):
            friendly_task_name = "-".join(partition_tasks)

            # remove any special characters from the task name that is not alphanumeric or hyphen
            friendly_task_name = re.sub(r"[^a-zA-Z0-9\-]+", "_", friendly_task_name)

            # replace multiple underscores or hyphens with a single hyphen
            friendly_task_name = re.sub(r"(\-_+|_+\-)", "-", friendly_task_name)

            # remove any leading/trailing '-' or '_'
            friendly_task_name = friendly_task_name.strip("_-")

            # truncate the task name if it is too long
            if len(friendly_task_name) > 50:
                h = md5(friendly_task_name.encode()).hexdigest()
                friendly_task_name = f"{friendly_task_name[:43]}-{h[:6]}"

            # set remote output directory
            remote_dir = f"{remote_output_prefix}/{dashboard}/{run_name}/{friendly_task_name}"
            flags.append(f"--remote-output-dir {remote_dir}")

        # set extra arguments
        flags.append(extra_args)

        # set compute gold bpb
        if compute_gold_bpb:
            flags.append(f"--task-args compute_gold_bpb=true")

        # set batch size
        if batch_size:
            flags.append(f"--batch-size {batch_size}")

        # set dry run
        if dry_run:
            flags.append("--dry-run")

        # set beaker image
        if beaker_image:
            flags.append(f"--beaker-image {beaker_image}")

        # set gantry args
        if use_gantry:
            flags.append(f"--gantry-args '{gantry_args}'")

        # run oe-eval
        subprocess.run(
            shlex.split(
                f"{OE_EVAL_LAUNCH_COMMAND} {' '.join(flags)} {'--dry-run' if dry_run else ''}"
            ),
            check=True,
            cwd=oe_eval_dir,
        )
        cnt += 1

    print(f"Launched {cnt:,} eval jobs on {beaker_clusters} for {run_name}.")


def convert_checkpoint(
    beaker_allow_dirty: bool,
    beaker_budget: str,
    beaker_cluster: str,
    beaker_dry_run: bool,
    beaker_gpus: int,
    beaker_priority: str,
    beaker_workspace: str,
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
    env: PythonEnv,
):
    if use_beaker:
        check_beaker_dependencies()

        assert input_dir.startswith("/"), "Input directory must be fully specified"
        if unsharded_output_dir:
            assert unsharded_output_dir.startswith(
                "/"
            ), "Unsharded output directory must be fully specified"
        if huggingface_output_dir:
            assert huggingface_output_dir.startswith(
                "/"
            ), "Huggingface output directory must be fully specified"

        weka_mounts = [
            mount
            for mount in (
                discover_weka_mount(input_dir),
                discover_weka_mount(unsharded_output_dir),
                discover_weka_mount(huggingface_output_dir),
            )
            if mount is not None
        ]

        gantry_flags = []

        for weka_path in set(weka_mounts):
            gantry_flags.append(f"--weka {weka_path}:/{weka_path}")

        if huggingface_token is not None:
            secret_name = add_secret_to_beaker_workspace(
                "HF_TOKEN", huggingface_token, beaker_workspace
            )
            if secret_name:
                gantry_flags.append(f"--env-secret HF_TOKEN={secret_name}")

        for cluster in BEAKER_KNOWN_CLUSTERS.get(beaker_cluster, [beaker_cluster]):
            gantry_flags.append(f"--cluster {cluster}")

        repo_root = find_repository_root()
        this_script_relative_to_repo = Path(__file__).relative_to(repo_root)

        remote_command = (
            f"python {this_script_relative_to_repo} "
            + f" --input-dir {input_dir} "
            + f" --olmo-type {olmo_type} "
            + (
                f" --huggingface-tokenizer {huggingface_tokenizer} "
                if huggingface_tokenizer
                else ""
            )
            + (f" --unsharded-output-dir {unsharded_output_dir} " if unsharded_output_dir else "")
            + (
                f" --huggingface-output-dir {huggingface_output_dir} "
                if huggingface_output_dir
                else ""
            )
            + f" --unsharded-output-suffix {unsharded_output_suffix} "
            + f" --huggingface-output-suffix {huggingface_output_suffix} "
            + f" --olmoe-commit-hash {olmoe_commit_hash} "
            + f" --olmo2-commit-hash {olmo2_commit_hash} "
        )

        gantry_command = (
            "gantry run "
            + f"--description 'Converting OLMo checkpoint at {input_dir}' "
            + ("--allow-dirty " if beaker_allow_dirty else "")
            + "--no-python "
            + f"--workspace {beaker_workspace} "
            + f"--priority {beaker_priority} "
            + f"--gpus {beaker_gpus} "
            + "--preemptible "
            + f"--budget {beaker_budget} "
            + "--yes "
            + ("--dry-run " if beaker_dry_run else "")
            + f"{' '.join(gantry_flags)} "
            + f"-- /bin/bash -c '{remote_command}'"
        )

        print(f"Submitting to beaker with command: {gantry_command}")
        return subprocess.run(shlex.split(gantry_command), check=True)

    remove_conflicting_packages()
    convert_olmo_checkpoint(
        type=olmo_type,
        input_dir=input_dir,
        unsharded_output_dir=unsharded_output_dir,
        huggingface_output_dir=huggingface_output_dir,
        huggingface_tokenizer=huggingface_tokenizer,
        unsharded_output_suffix=unsharded_output_suffix,
        huggingface_output_suffix=huggingface_output_suffix,
        env=env,
    )
