import json
import re
import shlex
import subprocess
import sys
from copy import deepcopy
from hashlib import md5
from typing import Optional
from urllib.parse import urlparse

from cookbook.cli.utils import (
    PythonEnv,
    add_aws_flags,
    add_secret_to_beaker_workspace,
    install_oe_eval,
    make_eval_run_name,
)
from cookbook.constants import (
    BEAKER_KNOWN_CLUSTERS,
    FIM_TOKENS,
    OE_EVAL_LAUNCH_COMMAND,
    WEKA_MOUNTS,
)
from cookbook.eval.named_tasks import NamedTasksGroupRegistry


def evaluate_checkpoint(
    oe_eval_commit: str,
    oe_eval_branch: str,
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
    gantry_args: str | dict,
    python_venv_name: str,
    python_venv_force: bool,
    vllm_memory_utilization: float,
    vllm_for_mc: bool,
    compute_gold_bpb: bool,
    model_args: Optional[dict],
    task_args: Optional[dict],
    fim_tokens: str,
    use_vllm_v1_spec: bool,
    use_backend_in_run_name: bool,
    name_suffix: str,
    num_shots: int | None,
):
    # Create virtual environment
    env = PythonEnv.create(name=python_venv_name, force=python_venv_force)
    print(f"Using Python virtual environment at {env.name}")

    # Install oe-eval toolkit
    oe_eval_dir = install_oe_eval(
        env=env,
        commit_hash=oe_eval_commit,
        commit_branch=oe_eval_branch,
        is_editable=use_gantry,
    )

    # this is where we store all fixed flags to pass to oe-eval
    flags: list[str] = []

    # clusters_to_exclude
    clusters_to_exclude: set[str] = set()

    # processing gantry/task args
    gantry_args_dict = json.loads(gantry_args.strip() or "{}") if isinstance(gantry_args, str) else gantry_args

    task_args_dict = json.loads(task_args.strip() or "{}") if isinstance(task_args, str) else (task_args or {})

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

    elif checkpoint_path.startswith("/weka/") or any(checkpoint_path.startswith(f"/{w}/") for w in WEKA_MOUNTS):
        print("Checkpoint is stored in Weka; I will remove cluster that have no WEKA.")
        for cl in BEAKER_KNOWN_CLUSTERS["goog"]:
            clusters_to_exclude.add(cl)

        if checkpoint_path.startswith("/weka/"):
            checkpoint_path = f"weka://{checkpoint_path[6:].rstrip('/')}"
        else:
            checkpoint_path = f"weka://{checkpoint_path.lstrip('/').rstrip('/')}"

    else:
        print("Path is a huggingface hub model; I will add huggingface token to beaker workspace")
        if huggingface_secret:
            hf_token_secret = add_secret_to_beaker_workspace(
                secret_name="HUGGING_FACE_HUB_TOKEN",
                secret_value=huggingface_secret,
                workspace=workspace,
                env=env,  # pyright: ignore
            )
            flags.append(f"--gantry-secret-hf-read-only '{hf_token_secret}'")
            gantry_args_dict = {"hf_token": "true", **gantry_args_dict}
        else:
            print("\n\nWARNING: Hugging Face token not provided; this may cause issues with model download.\n\n")

    if use_backend_in_run_name:
        if name_suffix.strip():
            raise ValueError("name_suffix and use_backend_in_run_name cannot both be provided")

        print("[WARNING] `--use-backend-in-run-name` is deprecated; use `--name-suffix <backend name>` instead.")
        name_suffix = model_backend

    # we use this run name when storing store all the output files
    run_name = make_eval_run_name(
        checkpoint_path=checkpoint_path,
        add_bos_token=add_bos_token,
        num_shots=num_shots,
        name_suffix=name_suffix.strip(),
    )

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

    # override number of shots; by default, the number of shots is part of task definition in oe-eval
    # changing number of shots will result in a new run name, similar to how we do it for BOS token.
    if num_shots is not None:
        flags.append(f"--num-shots {num_shots}")

    # figure out model args based on cli
    model_args = {
        "model_path": checkpoint_path,
        "add_bos_token": "true" if add_bos_token else "false",
        **({"gpu_memory_utilization": str(vllm_memory_utilization)} if model_backend == "vllm" else {}),
        **(model_args or {}),
    }
    model_args_str = ",".join(f"{k}={v}" for k, v in model_args.items())

    # set model info
    flags.append(f"--model {run_name}")
    flags.append(f"--model-args '{model_args_str}'")
    flags.append(f"--model-type {model_backend}")

    # these are all the tasks we want to run; note that we can't run regex patterns here,
    # they have to be actual strings
    all_tasks_set = set()
    for task_group in tasks:
        try:
            # this is a task group! the get function will return a class that has an expanded_tasks attribute
            all_tasks_set.update(NamedTasksGroupRegistry.get(task_group).expanded_tasks)
        except ValueError:
            # actually not a task group, just a task name. append as is.
            all_tasks_set.add(task_group)

    # we finish by sorting the tasks
    all_tasks = sorted(all_tasks_set)

    # @davidh: we have a few specific tasks that are not implemented in oe-eval as standalone tasks
    # @soldni: to clarify: this is fine, since these tasks are computed anyway as part of the non-bpb version,
    #          it's just the task alias that does not exist.
    EXCLUDE_FROM_LAUNCH = [
        r"^mmlu_.*:bpb::olmes$",
        r"^lambada:bpb$",
        r"^.*:pass_at_.*$",
    ]
    all_tasks = [task for task in all_tasks if not any(re.match(pattern, task) for pattern in EXCLUDE_FROM_LAUNCH)]

    # DOING SOME PRETTY PRINTING HERE #
    print(
        f"\n🏗️ Running following evals for \033[1m{run_name}\033[0m:",
        file=sys.stderr,
    )
    for task in all_tasks:
        print(f"  - {task}", file=sys.stderr)
    print(file=sys.stderr)
    # # # # # # # # # # # # # # # # # #

    # we need to partition tasks based on whether they are mc, gen, or rc
    partitioned_tasks = {}
    for task in all_tasks:
        if ":rc::" in task:
            partitioned_tasks.setdefault("rc", []).append(task)
        elif ":mc::" in task:
            partitioned_tasks.setdefault("mc", []).append(task)
        elif "fim_" in task:
            partitioned_tasks.setdefault("fim", []).append(task)
        # TODO: automatically partition HF tasks --@soldni
        # elif task in {"ultrachat_masked_ppl", "wildchat_masked_ppl"}:
        #     # these tasks don't work with vllm, so we run them on huggingface
        #     partitioned_tasks.setdefault("hf", []).append(task)
        else:
            partitioned_tasks.setdefault("gen", []).append(task)

    # we launch jobs by partition. We are careful to partition RC/MC/GEN tasks separately
    submitted_jobs_cnt = 0
    for task_group, tasks_names in partitioned_tasks.items():
        for i in range(0, len(tasks_names), partition_size or len(tasks_names)):
            local_flags = deepcopy(flags)
            partition_task_args = deepcopy(task_args_dict)

            # add all tasks in the partition as flag
            partition_tasks = tasks_names[i : i + partition_size] if partition_size else tasks_names
            escaped_partition_tasks = [
                json.dumps(task) if task[0] == "{" else task.replace(" ", "\\ ") for task in partition_tasks
            ]

            local_flags.append(f"--task {' '.join(escaped_partition_tasks)}")

            if add_aws_flags(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                workspace=workspace,
                flags=local_flags,
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

            # set revision
            if revision:
                local_flags.append(f"--revision {revision}")

            # set gantry
            if use_gantry:
                local_flags.append("--use-gantry")

            if beaker_retries > 0:
                local_flags.append(f"--beaker-retries {beaker_retries}")

            # user might want to disable vllm v1 spec because its causing eval failures
            gantry_args_dict = {
                "env": f"VLLM_USE_V1={1 if use_vllm_v1_spec else 0}",
                **gantry_args_dict,
            }

            # finally append gantry args
            local_flags.append(f"--gantry-args '{json.dumps(gantry_args_dict)}'")

            if model_backend == "vllm" and task_group == "mc" and vllm_for_mc:
                local_flags.append("--vllm-for-mc")

            if fim_tokens and task_group == "fim":
                infilling_dict = FIM_TOKENS[fim_tokens]

                # Add FIM tokens to context, preserving other existing kwargs
                partition_task_args.setdefault("context_kwargs", {})
                partition_task_args["context_kwargs"].update(infilling_dict["context_kwargs"])

                # Add FIM stop sequences, preserving other existing stop sequences
                partition_task_args.setdefault("generation_kwargs", {})
                if "stop_sequences" in partition_task_args["generation_kwargs"]:
                    # Add the stop tokens if they do not exist
                    partition_task_args["generation_kwargs"]["stop_sequences"].extend(
                        [
                            stop_tok
                            for stop_tok in infilling_dict["generation_kwargs"]["stop_sequences"]
                            if stop_tok not in partition_task_args["generation_kwargs"]["stop_sequences"]
                        ]
                    )
                else:
                    partition_task_args["generation_kwargs"].update(infilling_dict["generation_kwargs"])

            if compute_gold_bpb:
                partition_task_args["compute_gold_bpb"] = True

            if partition_task_args:
                local_flags.append(f"--task-args '{json.dumps(partition_task_args)}'")

            # run oe-eval
            cmd = f"{env.python} {OE_EVAL_LAUNCH_COMMAND} {' '.join(local_flags)}"
            print(f"\n\nCommand:\n{cmd}\nFrom:\n{oe_eval_dir}\n\n")
            output = subprocess.run(shlex.split(cmd), cwd=oe_eval_dir, env=env.path(), capture_output=True)
            print(f"{output.stdout.decode()}\n{output.stderr.decode()}\n")
            if output.returncode != 0:
                raise RuntimeError(f"Error running command: {cmd}")
            submitted_jobs_cnt += 1

    print(f"Launched {submitted_jobs_cnt:,} eval jobs on {beaker_clusters} for {run_name}.")
