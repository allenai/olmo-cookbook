import os
import shlex
import shutil
import subprocess
from typing import Optional

from cookbook.cli.utils import (
    PythonEnv,
    add_secret_to_beaker_workspace,
    discover_weka_mount,
    install_beaker_py,
    install_olmo_core,
    install_transformers,
    make_destination_dir,
    remove_conflicting_packages,
)
from cookbook.constants import (
    BEAKER_KNOWN_CLUSTERS,
    OLMO_CORE_CONVERT_FROM_HF_SCRIPT,
    OLMO_CORE_V2_COMMIT_HASH,
    TRANSFORMERS_COMMIT_HASH,
)


def convert_hf_to_olmo_core_v2(
    input_dir: str,
    output_dir: Optional[str] = None,
    output_suffix: str = "olmo_core",
    olmo_core_v2_commit_hash: str = OLMO_CORE_V2_COMMIT_HASH,
    olmo_core_v2_experiment_json_path: Optional[str] = None,
    olmo_core_v2_model_arch: Optional[str] = None,
    olmo_core_v2_tokenizer: Optional[str] = None,
    transformers_git_url: Optional[str] = None,
    transformers_commit_hash: str = TRANSFORMERS_COMMIT_HASH,
    transformers_model_id: Optional[str] = None,
    transformers_revision: str = "main",
    skip_validation: bool = False,
    debug_validation: bool = False,
    device: Optional[str] = None,
    env: Optional[PythonEnv] = None,
):
    env = env or PythonEnv.null()

    directories_to_clean_up = []

    output_dir = make_destination_dir(input_dir, output_suffix, output_dir)

    try:
        print("Starting conversion of HF model...")

        olmo_code_dir = install_olmo_core(env=env, commit_hash=olmo_core_v2_commit_hash)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(transformers_commit_hash, env, git_url=transformers_git_url)
        directories_to_clean_up.append(huggingface_code_dir)

        print("Converting Huggingface weights to OLMo core V2 format...")
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            env.python,
            OLMO_CORE_CONVERT_FROM_HF_SCRIPT,
            f"--checkpoint-input-path {input_dir}",
            f"--output-dir {output_dir}",
            f"--revision {transformers_revision}",
            (f"--config-path {olmo_core_v2_experiment_json_path}" if olmo_core_v2_experiment_json_path else ""),
            (f"--model-arch {olmo_core_v2_model_arch}" if olmo_core_v2_model_arch else ""),
            (f"--tokenizer {olmo_core_v2_tokenizer}" if olmo_core_v2_tokenizer else ""),
            (f"--model-id {transformers_model_id}" if transformers_model_id else ""),
            (f"--device {device}" if device else ""),
            ("--skip-validation" if skip_validation else ""),
            ("--debug" if debug_validation else ""),
        ]
        print(f"Running command: {' '.join(cmd)} from commit hash: {olmo_core_v2_commit_hash}")

        try:
            subprocess.run(
                shlex.split(" ".join(cmd)),
                check=True,
                cwd=olmo_code_dir,
                env=env.path(),
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Conversion failed with output: {e.output}") from e

        print(f"Completed conversion of HF model. OLMo core v2 model at {output_dir}.")

    finally:
        for directory in directories_to_clean_up:
            print(f"Cleaning up {directory}...")
            shutil.rmtree(directory, ignore_errors=True)


def run_checkpoint_conversion_from_hf(
    beaker_allow_dirty: bool,
    beaker_budget: str,
    beaker_cluster: str,
    beaker_dry_run: bool,
    beaker_gpus: int,
    beaker_priority: str,
    beaker_workspace: str,
    beaker_preemptible: bool,
    huggingface_token: Optional[str],
    input_dir: str,
    output_dir: Optional[str],
    output_suffix: str,
    olmo_core_v2_commit_hash: str,
    olmo_core_v2_experiment_json_path: Optional[str],
    olmo_core_v2_model_arch: Optional[str],
    olmo_core_v2_tokenizer: Optional[str],
    huggingface_transformers_git_url: Optional[str],
    huggingface_transformers_commit_hash: str,
    huggingface_transformers_model_id: Optional[str],
    huggingface_transformers_revision: str,
    use_beaker: bool,
    use_system_python: bool,
    python_venv_name: str,
    python_venv_force: bool,
    skip_validation: bool,
    debug_validation: bool,
    torch_device: Optional[str],
):
    env = (
        PythonEnv.create(name=python_venv_name, force=python_venv_force)
        if not use_system_python
        else PythonEnv.null()
    )

    if use_beaker:
        print("Installing beaker and gantry clients...")
        install_beaker_py(env=env)

        assert input_dir.startswith("/"), "Input directory must be fully specified"
        if output_dir:
            assert output_dir.startswith("/"), "Output directory must be fully specified"
        if olmo_core_v2_experiment_json_path:
            assert olmo_core_v2_experiment_json_path.startswith("/"), "Output directory must be fully specified"

        weka_mounts = [
            mount
            for mount in (
                discover_weka_mount(input_dir),
                discover_weka_mount(output_dir),
            )
            if mount is not None
        ]

        gantry_flags = []

        for weka_path in set(weka_mounts):
            gantry_flags.append(f"--weka {weka_path}:/{weka_path}")

        if huggingface_token is not None:
            secret_name = add_secret_to_beaker_workspace(
                secret_name="HF_TOKEN",
                secret_value=huggingface_token,
                workspace=beaker_workspace,
                env=env,  # type: ignore
            )
            if secret_name:
                gantry_flags.append(f"--env-secret HF_TOKEN={secret_name}")

        for cluster in BEAKER_KNOWN_CLUSTERS.get(beaker_cluster, [beaker_cluster]):
            gantry_flags.append(f"--cluster {cluster}")

        remote_command = [
            "pip install uv && uv pip install . --system &&",
            "olmo-cookbook-eval convert-from-hf",
            f"{input_dir}",
            (f"--output-dir {output_dir}" if output_dir else ""),
            f"--output-suffix {output_suffix}",
            f"--olmo-core-v2-commit-hash {olmo_core_v2_commit_hash}",
            (
                f"--olmo-core-v2-experiment-json-path {olmo_core_v2_experiment_json_path}"
                if olmo_core_v2_experiment_json_path
                else ""
            ),
            (f"--olmo-core-v2-model-arch {olmo_core_v2_model_arch}" if olmo_core_v2_model_arch else ""),
            (f"--olmo-core-v2-tokenizer {olmo_core_v2_tokenizer}" if olmo_core_v2_tokenizer else ""),
            f"--huggingface-transformers-git-url {huggingface_transformers_git_url}",
            f"--huggingface-transformers-commit-hash {huggingface_transformers_commit_hash}",
            (
                f"--huggingface-transformers-model-id {huggingface_transformers_model_id}"
                if huggingface_transformers_model_id
                else ""
            ),
            f"--huggingface-transformers-revision {huggingface_transformers_revision}",
            "--use-system-python",
            ("--skip-validation" if skip_validation else ""),
            ("--debug-validation" if debug_validation else ""),
            ("--torch-device" if torch_device else ""),
        ]
        remote_command_str = " ".join(remote_command)

        gantry_command = [
            "gantry run",
            f"--description 'Converting HF checkpoint at {input_dir}'",
            ("--allow-dirty" if beaker_allow_dirty else ""),
            "--no-python",
            f"--workspace {beaker_workspace}",
            f"--priority {beaker_priority}",
            f"--gpus {beaker_gpus}",
            ("--preemptible" if beaker_preemptible else ""),
            f"--budget {beaker_budget}",
            "--yes",
            ("--dry-run" if beaker_dry_run else ""),
            " ".join(gantry_flags),
            f"-- /bin/bash -c '{remote_command_str}'",
        ]
        gantry_command_str = " ".join(gantry_command)

        print(f"Submitting to beaker with command: {gantry_command_str}")
        return subprocess.run(shlex.split(gantry_command_str), check=True, env=env.path())

    remove_conflicting_packages(env=env)

    return convert_hf_to_olmo_core_v2(
        input_dir=input_dir,
        output_dir=output_dir,
        output_suffix=output_suffix,
        olmo_core_v2_commit_hash=olmo_core_v2_commit_hash,
        transformers_git_url=huggingface_transformers_git_url,
        transformers_commit_hash=huggingface_transformers_commit_hash,
        transformers_revision=huggingface_transformers_revision,
        skip_validation=skip_validation,
        debug_validation=debug_validation,
        device=torch_device,
        env=env,
    )
