from copy import deepcopy
import re
import shlex
import subprocess
from hashlib import md5
from urllib.parse import urlparse

from cookbook.cli.utils import (
    PythonEnv,
    add_aws_flags,
    add_secret_to_beaker_workspace,
    install_oe_eval,
    make_eval_run_name,
)
from cookbook.constants import (
    ALL_NAMED_GROUPS,
    BEAKER_KNOWN_CLUSTERS,
    OE_EVAL_LAUNCH_COMMAND,
    WEKA_MOUNTS,
)


# /Users/lucas/Code/olmo-cookbook/.venv/oe-eval-venv/bin/python oe_eval/launch.py
#   --beaker-workspace 'ai2/lm-eval'
#   --beaker-budget 'ai2/oe-data'
#   --beaker-priority high
#   --cluster 'ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale,ai2/saturn-cirrascale'
#   --gpus 2
#   --datalake-tags 'dashboard=peteish32,checkpoint=peteish32-from716000-peteish32-from716000-100Banneal_step11921-hf'
#   --push-datalake
#   --model peteish32-from716000-peteish32-from716000-100Banneal_step11921-hf
#   --model-args 'model_path=weka://oe-training-default/ai2-llm/checkpoints/peteish32-anneal/peteish32-from716000-peteish32-from716000-100Banneal/step11921-hf,add_bos_token=False'
#   --model-type vllm
#   --task coqa::olmes drop::olmes
#   --gantry-secret-aws-access-key-id 'lucas_AWS_ACCESS_KEY_ID'
#   --gantry-secret-aws-secret-access 'lucas_AWS_SECRET_ACCESS_KEY'
#   --remote-output-dir s3://ai2-llm/evaluation/peteish32/peteish32-from716000-peteish32-from716000-100Banneal_step11921-hf/coqa_olmes-drop_olmes
#   --task-args compute_gold_bpb=true
#   --model-args gpu_memory_utilization=0.6
#   --use-gantry
#   --beaker-image lucas/vllm-0.7.2patch-olmo2-32b
#   --task gsm8k::olmo1 naturalqs::olmes
#   --remote-output-dir s3://ai2-llm/evaluation/peteish32/peteish32-from716000-peteish32-from716000-100Banneal_step11921-hf/gsm8k_olmo1-naturalqs_olmes
#   --task-args compute_gold_bpb=true
#   --model-args gpu_memory_utilization=0.6
#   --use-gantry
#   --beaker-image lucas/vllm-0.7.2patch-olmo2-32b
#   --task squad::olmes
#   --remote-output-dir s3://ai2-llm/evaluation/peteish32/peteish32-from716000-peteish32-from716000-100Banneal_step11921-hf/squad_olmes
#   --task-args compute_gold_bpb=true
#   --model-args gpu_memory_utilization=0.6
#   --use-gantry
#   --beaker-image lucas/vllm-0.7.2patch-olmo2-32b


def evaluate_checkpoint(
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
    python_venv_name: str,
    python_venv_force: bool,
):
    # Create virtual environment
    env = PythonEnv.create(name=python_venv_name, force=python_venv_force)
    print(f"Using Python virtual environment at {env.name}")

    # Install oe-eval toolkit
    use_gantry_when_launching_eval = "--use-gantry" in extra_args
    oe_eval_dir = install_oe_eval(
        env=env,
        commit_hash=oe_eval_commit,
        is_editable=use_gantry_when_launching_eval,
        # no_dependencies=(not use_gantry_when_launching_eval),
    )

    # this is where we store all fixed flags to pass to oe-eval
    flags: list[str] = []

    # clusters_to_exclude
    clusters_to_exclude: set[str] = set()

    # Need to figure out how checkpoint is stored!
    if (scheme := urlparse(checkpoint_path).scheme) == "s3":
        print("Checkpoint is stored in S3; I will add AWS credentials to workspace and pass them to oe-eval")
        is_aws_cred_added = add_aws_flags(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            workspace=workspace,
            flags=flags,
            env=env,
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
    elif checkpoint_path.startswith("/") and any(re.match(rf"/{w}/", checkpoint_path) for w in WEKA_MOUNTS):
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
    beaker_clusters = ",".join(set(BEAKER_KNOWN_CLUSTERS.get(cluster, [cluster])).difference(clusters_to_exclude))

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
    flags.append(f"--model {run_name}")
    flags.append(f"--model-args 'model_path={checkpoint_path},add_bos_token={add_bos_token}'")
    flags.append(f"--model-type {model_backend}")

    all_tasks = sorted(
        set(task for task_group in tasks for task in ALL_NAMED_GROUPS.get(task_group, [task_group]))
    )

    cnt = 0
    for i in range(0, len(all_tasks), partition_size or len(all_tasks)):
        local_flags = deepcopy(flags)

        # add all tasks in the partition as flag
        partition_tasks = all_tasks[i: i + partition_size] if partition_size else all_tasks
        flags.append(f"--task {' '.join(partition_tasks)}")

        if add_aws_flags(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            workspace=workspace,
            flags=flags,
            env=env,
        ):
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
            local_flags.append(f"--remote-output-dir '{remote_dir}'")

        # set extra arguments
        local_flags.append(extra_args)

        # set batch size
        if batch_size:
            local_flags.append(f"--batch-size {batch_size}")

        # set dry run
        if dry_run:
            local_flags.append("--dry-run")

        # set beaker image
        if beaker_image:
            local_flags.append(f"--beaker-image {beaker_image}")

        # set gantry args
        if use_gantry:
            local_flags.append(f"--gantry-args '{gantry_args}'")

        # run oe-eval
        cmd = f"{env.python} {OE_EVAL_LAUNCH_COMMAND} {' '.join(local_flags)}"
        print(f"\n\nCommand:\n{cmd}\nFrom:\n{oe_eval_dir}\n\n")
        subprocess.run(shlex.split(cmd), check=True, cwd=oe_eval_dir, env=env.path())
        cnt += 1

    print(f"Launched {cnt:,} eval jobs on {beaker_clusters} for {run_name}.")
