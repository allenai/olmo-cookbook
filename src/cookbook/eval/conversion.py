import json
import os
import shlex
import shutil
import subprocess
from typing import Optional

import yaml

from cookbook.cli.utils import (
    PythonEnv,
    add_secret_to_beaker_workspace,
    discover_weka_mount,
    download_tokenizer,
    install_beaker_py,
    install_olmo,
    install_olmo_core,
    install_transformers,
    make_destination_dir,
    remove_conflicting_packages,
)
from cookbook.constants import (
    DEFAULT_OLMO2_TOKENIZER,
    DEFAULT_OLMO_CORE_TOKENIZER,
    DEFAULT_OLMOE_TOKENIZER,
    OLMO2_COMMIT_HASH,
    OLMO2_CONVERSION_SCRIPT,
    OLMO2_UNSHARD_SCRIPT,
    OLMO_CORE_COMMIT_HASH,
    OLMO_CORE_UNSHARD_CONVERT_SCRIPT,
    OLMO_CORE_V2_COMMIT_HASH,
    OLMOE_CONVERSION_SCRIPT,
    OLMOE_UNSHARD_SCRIPT,
    TRANSFORMERS_COMMIT_HASH,
)
from cookbook.utils.clusters import get_matching_clusters


def convert_olmo_core_v2(
    input_dir: str,
    huggingface_tokenizer: str = DEFAULT_OLMO_CORE_TOKENIZER,
    max_sequence_length: Optional[int] = None,
    unsharded_output_dir: Optional[str] = None,
    huggingface_output_dir: Optional[str] = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
    olmo_core_v2_commit_hash: str = OLMO_CORE_V2_COMMIT_HASH,
    transformers_git_url: Optional[str] = None,
    transformers_commit_hash: str = TRANSFORMERS_COMMIT_HASH,
    skip_validation: bool = False,
    fix_generation_config: bool = True,
    dtype: Optional[str] = None,
    env: Optional[PythonEnv] = None,
):
    env = env or PythonEnv.null()

    directories_to_clean_up = []

    if max_sequence_length is None:
        # we load config.json from the input_dir and get the max_sequence_length from the config
        config_file = os.path.join(input_dir, "config.json")
        assert os.path.exists(config_file), f"Could not find config.json in {input_dir}"

        with open(config_file, "r") as f:
            config = json.load(f)
        max_sequence_length = config.get("dataset", {}).get("sequence_length", None)

        if max_sequence_length is None:
            raise ValueError(
                f"Could not find max_sequence_length in config.json for {input_dir}; "
                "please provide it as a command line argument."
            )

    unsharded_output_dir = make_destination_dir(input_dir, unsharded_output_suffix, unsharded_output_dir)
    huggingface_output_dir = make_destination_dir(input_dir, huggingface_output_suffix, huggingface_output_dir)

    try:
        print("Starting conversion of OLMo core model...")

        olmo_code_dir = install_olmo_core(env=env, commit_hash=olmo_core_v2_commit_hash)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(transformers_commit_hash, env, git_url=transformers_git_url)
        directories_to_clean_up.append(huggingface_code_dir)

        tokenizer_dir = download_tokenizer(huggingface_tokenizer, env)
        directories_to_clean_up.append(tokenizer_dir)

        # copy all tokenizer files to the huggingface output dir
        #     we copy the tokenizer first because it might carry over config.json from the model we are
        #     grabbing tokenizer from (e.g. allenai/OLMo-2-1124-7B), which might be incorrect for the model
        #     we are converting. but that's okay if we copy the tokenizer first, cuz we will override it
        #     after. This was an awful bug, and took me (@soldni) forever to figure out. Embarrassing.
        for file in os.listdir(tokenizer_dir):
            if not os.path.isfile(src := os.path.join(tokenizer_dir, file)):
                # do not copy directories
                continue
            if file.startswith("."):
                # do not copy hidden files
                continue
            shutil.copy(src, os.path.join(huggingface_output_dir, file))
        print(f"Copied tokenizer files to {huggingface_output_dir}.")

        # check if input_dir contains a "config.yaml" file. if it does not, check if it contains a
        # "config.json" file. if it does, re-save it as a "config.yaml" file.
        config_file = os.path.join(input_dir, "config.yaml")
        if not os.path.exists(config_file):
            config_json_file = os.path.join(input_dir, "config.json")
            if not os.path.exists(config_json_file):
                raise FileNotFoundError(f"Could not find 'config.yaml' or 'config.json' in {input_dir}")

            print("Converting 'config.json' to 'config.yaml'...")
            with open(config_json_file, "r") as json_file, open(config_file, "w") as yaml_file:
                config = json.load(json_file)
                yaml.dump(config, yaml_file)

        print("Converting OLMo core V2 weights to Huggingface format...")
        os.makedirs(huggingface_output_dir, exist_ok=True)
        cmd = [
            env.python,
            OLMO_CORE_UNSHARD_CONVERT_SCRIPT,
            "--checkpoint-input-path",
            input_dir,
            "--max-sequence-length",
            str(max_sequence_length),
            "--huggingface-output-dir",
            huggingface_output_dir,
            ("--skip-validation" if skip_validation else ""),
            (f"--dtype {dtype}" if dtype else ""),
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

        print(f"Completed conversion of OLMo core V2 checkpoint. Huggingface model at {huggingface_output_dir}.")

        # change type of model to bfloat16 if it is float32
        config_file = os.path.join(huggingface_output_dir, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)

        if config.get("torch_dtype", "") == "float32":
            print("Changing type of model to bfloat16...")
            config["torch_dtype"] = "bfloat16"

            with open(config_file, "w") as f:
                json.dump(config, f)

            print(f"Updated model type from float32 to bfloat16 in {config_file}.")

        if fix_generation_config:
            # fix generation config
            generation_config_file = os.path.join(huggingface_output_dir, "generation_config.json")
            tokenization_config_file = os.path.join(huggingface_output_dir, "tokenizer_config.json")
            tokenizer_file = os.path.join(huggingface_output_dir, "tokenizer.json")

            generation_config = {}
            if os.path.exists(generation_config_file):
                with open(generation_config_file, "r") as f:
                    generation_config = json.load(f)

            # @soldni: this is to stop type checking from complaining
            assert isinstance(generation_config, dict), "generation_config.json should be a dictionary"

            if "eos_token_id" in generation_config and "pad_token_id" not in generation_config:
                print("EOS and PAD tokens already set in generation config, nothing to do.")
                return

            if not os.path.exists(tokenization_config_file):
                raise FileNotFoundError("tokenization_config.json not found; cannot fix generation config.")

            with open(tokenization_config_file, "r") as f:
                tokenization_config = json.load(f)

            if not os.path.exists(tokenizer_file):
                raise FileNotFoundError("tokenizer.json not found; cannot fix generation config.")

            with open(tokenizer_file, "r") as f:
                tokenizer = json.load(f)

            # TODO: bos too?
            for token_name in ("eos_token", "pad_token"):
                token_id_name = f"{token_name}_id"
                if token_id_name in generation_config:
                    print(f"Checking EOS token ID in generation config: {generation_config['eos_token_id']}")
                    continue

                if (token_value := tokenization_config.get(token_name)) is None:
                    continue

                token_id_value = tokenizer.get("model", {}).get("vocab", {}).get(token_value, None)
                if token_id_value is None:
                    raise ValueError(f"Could not find {token_id_name} for {token_value} in {tokenizer_file}.")
                print(f"Setting {token_id_name} to {token_id_value} in generation config.")
                generation_config[token_id_name] = token_id_value

            with open(generation_config_file, "w") as f:
                json.dump(generation_config, f, indent=2)

            print(f"Updated generation config in {generation_config_file}.")

    finally:
        for directory in directories_to_clean_up:
            print(f"Cleaning up {directory}...")
            shutil.rmtree(directory, ignore_errors=True)


def convert_olmo_core(
    input_dir: str,
    huggingface_tokenizer: str = DEFAULT_OLMO_CORE_TOKENIZER,
    unsharded_output_dir: Optional[str] = None,
    huggingface_output_dir: Optional[str] = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
    olmo_commit_hash: str = OLMO_CORE_COMMIT_HASH,
    transformers_git_url: Optional[str] = None,
    transformers_commit_hash: str = TRANSFORMERS_COMMIT_HASH,
    env: Optional[PythonEnv] = None,
):
    env = env or PythonEnv.null()

    current_directory = os.getcwd()
    directories_to_clean_up = []

    unsharded_output_dir = make_destination_dir(input_dir, unsharded_output_suffix, unsharded_output_dir)
    huggingface_output_dir = make_destination_dir(input_dir, huggingface_output_suffix, huggingface_output_dir)

    try:
        print("Starting conversion of OLMo core model...")

        olmo_code_dir = install_olmo_core(env=env, commit_hash=olmo_commit_hash)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(transformers_commit_hash, env, git_url=transformers_git_url)
        directories_to_clean_up.append(huggingface_code_dir)

        tokenizer_dir = download_tokenizer(huggingface_tokenizer, env)
        directories_to_clean_up.append(tokenizer_dir)

        # check if input_dir contains a "config.yaml" file. if it does not, check if it contains a
        # "config.json" file. if it does, re-save it as a "config.yaml" file.
        config_file = os.path.join(input_dir, "config.yaml")
        if not os.path.exists(config_file):
            config_json_file = os.path.join(input_dir, "config.json")
            if not os.path.exists(config_json_file):
                raise FileNotFoundError(f"Could not find 'config.yaml' or 'config.json' in {input_dir}")

            print("Converting 'config.json' to 'config.yaml'...")
            with open(config_json_file, "r") as json_file, open(config_file, "w") as yaml_file:
                config = json.load(json_file)
                yaml.dump(config, yaml_file)

        print("Converting OLMo core weights to Huggingface format...")
        os.makedirs(huggingface_output_dir, exist_ok=True)
        cmd = [
            env.python,
            OLMO_CORE_UNSHARD_CONVERT_SCRIPT,
            "--checkpoint-input-dir",
            input_dir,
            "--unsharded-output-dir",
            unsharded_output_dir,
            "--huggingface-output-dir",
            huggingface_output_dir,
            "--tokenizer-name-or-path",
            huggingface_tokenizer,
        ]
        subprocess.run(shlex.split(" ".join(cmd)), check=True, cwd=olmo_code_dir, env=env.path())
        print(f"Conversion of OLMo core checkpoint complete. Huggingface model saved to {huggingface_output_dir}.")

    except Exception as e:
        print(f"Error cloning repositories: {e}")
        os.chdir(current_directory)
        raise e
    finally:
        for directory in directories_to_clean_up:
            print(f"Cleaning up {directory}...")
            shutil.rmtree(directory, ignore_errors=True)


def convert_olmoe(
    input_dir: str,
    huggingface_tokenizer: str = DEFAULT_OLMOE_TOKENIZER,
    unsharded_output_dir: Optional[str] = None,
    huggingface_output_dir: Optional[str] = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
    olmo_commit_hash: Optional[str] = None,
    transformers_git_url: Optional[str] = None,
    transformers_commit_hash: Optional[str] = None,
    env: Optional[PythonEnv] = None,
):
    env = env or PythonEnv.null()

    current_directory = os.getcwd()
    directories_to_clean_up = []

    unsharded_output_dir = make_destination_dir(input_dir, unsharded_output_suffix, unsharded_output_dir)
    huggingface_output_dir = make_destination_dir(input_dir, huggingface_output_suffix, huggingface_output_dir)

    try:
        print("Starting conversion of OLMoE model...")

        olmo_code_dir = install_olmo(olmo_commit_hash, env)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(transformers_commit_hash, env, git_url=transformers_git_url)
        directories_to_clean_up.append(huggingface_code_dir)

        model_dir = os.path.join(input_dir, "model")
        is_model_in_directory = os.path.exists(model_dir) and os.path.isdir(model_dir)

        tokenizer_dir = download_tokenizer(huggingface_tokenizer, env)
        directories_to_clean_up.append(tokenizer_dir)

        if is_model_in_directory:
            print("OLMo checkpoint is sharded. Unsharding...")
            unshard_command = [
                env.python,
                OLMOE_UNSHARD_SCRIPT,
                f"'{input_dir}'",
                f"'{unsharded_output_dir}'",
                "--model-only",
            ]
            os.makedirs(unsharded_output_dir, exist_ok=True)
            subprocess.run(
                shlex.split(" ".join(unshard_command)),
                check=True,
                cwd=olmo_code_dir,
                env=env.path(),
            )
            print(f"Unsharding to {unsharded_output_dir} complete.")
        else:
            print("OLMo checkpoint is not sharded. Skipping unsharding...")
            shutil.rmtree(unsharded_output_dir, ignore_errors=True)
            unsharded_output_dir = input_dir

        print("Converting OLMoE weights to Huggingface format...")
        os.makedirs(huggingface_output_dir, exist_ok=True)
        cmd = [
            env.python,
            OLMOE_CONVERSION_SCRIPT,
            "--input_dir",
            unsharded_output_dir,
            "--tokenizer_json_path",
            os.path.join(tokenizer_dir, "tokenizer.json"),
            "--output_dir",
            huggingface_output_dir,
        ]
        subprocess.run(shlex.split(" ".join(cmd)), check=True, cwd=huggingface_code_dir, env=env.path())
        print(f"Conversion of OLMoE checkpoint complete. Huggingface model saved to {huggingface_output_dir}.")

    except Exception as e:
        print(f"Error: {e}")
        os.chdir(current_directory)
        raise e
    finally:
        for directory in directories_to_clean_up:
            print(f"Cleaning up {directory}...")
            shutil.rmtree(directory, ignore_errors=True)


def convert_olmo2(
    input_dir: str,
    huggingface_tokenizer: str = DEFAULT_OLMO2_TOKENIZER,
    unsharded_output_dir: Optional[str] = None,
    huggingface_output_dir: Optional[str] = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
    olmo_commit_hash: str = OLMO2_COMMIT_HASH,
    transformers_git_url: Optional[str] = None,
    transformers_commit_hash: str = TRANSFORMERS_COMMIT_HASH,
    env: Optional[PythonEnv] = None,
):
    env = env or PythonEnv.null()
    current_directory = os.getcwd()
    directories_to_clean_up = []

    unsharded_output_dir = make_destination_dir(input_dir, unsharded_output_suffix, unsharded_output_dir)
    huggingface_output_dir = make_destination_dir(input_dir, huggingface_output_suffix, huggingface_output_dir)

    try:
        print("Starting conversion of OLMo 2 model...")

        olmo_code_dir = install_olmo(olmo_commit_hash, env)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(transformers_commit_hash, env, git_url=transformers_git_url)
        directories_to_clean_up.append(huggingface_code_dir)

        model_dir = os.path.join(input_dir, "model")
        is_model_in_directory = os.path.exists(model_dir) and os.path.isdir(model_dir)

        tokenizer_dir = download_tokenizer(huggingface_tokenizer, env)
        directories_to_clean_up.append(tokenizer_dir)

        if is_model_in_directory:
            print("OLMo checkpoint is sharded. Unsharding...")
            unshard_command = [
                env.python,
                OLMO2_UNSHARD_SCRIPT,
                f"'{input_dir}'",
                f"'{unsharded_output_dir}'",
                "--model-only",
            ]
            os.makedirs(unsharded_output_dir, exist_ok=True)
            subprocess.run(
                shlex.split(" ".join(unshard_command)),
                check=True,
                cwd=olmo_code_dir,
                env=env.path(),
            )
            print(f"Unsharding to {unsharded_output_dir} complete.")
        else:
            print("OLMo checkpoint is not sharded. Skipping unsharding...")
            shutil.rmtree(unsharded_output_dir, ignore_errors=True)
            unsharded_output_dir = input_dir

        print("Converting OLMo 2 weights to Huggingface format...")
        os.makedirs(huggingface_output_dir, exist_ok=True)
        cmd = [
            env.python,
            OLMO2_CONVERSION_SCRIPT,
            "--input_dir",
            unsharded_output_dir,
            "--tokenizer_json_path",
            os.path.join(tokenizer_dir, "tokenizer.json"),
            "--output_dir",
            huggingface_output_dir,
        ]
        subprocess.run(shlex.split(" ".join(cmd)), check=True, cwd=huggingface_code_dir, env=env.path())
        print(f"Conversion of OLMo 2 checkpoint complete. Huggingface model saved to {huggingface_output_dir}.")

    except Exception as e:
        print(f"Error cloning repositories: {e}")
        os.chdir(current_directory)
        raise e
    finally:
        for directory in directories_to_clean_up:
            print(f"Cleaning up {directory}...")
            shutil.rmtree(directory, ignore_errors=True)


def run_checkpoint_conversion(
    beaker_allow_dirty: bool,
    beaker_budget: str,
    beaker_cluster: str,
    beaker_dry_run: bool,
    beaker_gpus: int,
    beaker_priority: str,
    beaker_workspace: str,
    beaker_preemptible: bool,
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
    huggingface_transformers_git_url: Optional[str],
    huggingface_transformers_commit_hash: str,
    unsharded_output_dir: Optional[str],
    unsharded_output_suffix: str,
    use_beaker: bool,
    use_system_python: bool,
    python_venv_name: str,
    python_venv_force: bool,
    max_sequence_length: Optional[int] = None,
    skip_validation: bool = False,
    dtype: Optional[str] = None,
    experimental_with_flash_attn: bool = False,
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
        if unsharded_output_dir:
            assert unsharded_output_dir.startswith("/"), "Unsharded output directory must be fully specified"
        if huggingface_output_dir:
            assert huggingface_output_dir.startswith("/"), "Huggingface output directory must be fully specified"

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
                secret_name="HF_TOKEN",
                secret_value=huggingface_token,
                workspace=beaker_workspace,
                env=env,  # type: ignore
            )
            if secret_name:
                gantry_flags.append(f"--env-secret HF_TOKEN={secret_name}")

        for cluster in get_matching_clusters(beaker_cluster):
            gantry_flags.append(f"--cluster {cluster}")

        install_flash_attention = ""
        if experimental_with_flash_attn and olmo_type == "olmo-core-v2":
            install_flash_attention = "pip install --no-build-isolation-package flash-attn &&"

        remote_command = [
            "pip install uv && uv pip install . --system &&",
            install_flash_attention,
            "olmo-cookbook-eval convert",
            f"{input_dir}",
            f"--olmo-type {olmo_type}",
            (f"--huggingface-tokenizer {huggingface_tokenizer}" if huggingface_tokenizer else ""),
            (f"--unsharded-output-dir {unsharded_output_dir}" if unsharded_output_dir else ""),
            (f"--huggingface-output-dir {huggingface_output_dir}" if huggingface_output_dir else ""),
            f"--unsharded-output-suffix {unsharded_output_suffix}",
            f"--huggingface-output-suffix {huggingface_output_suffix}",
            f"--olmoe-commit-hash {olmoe_commit_hash}",
            f"--olmo2-commit-hash {olmo2_commit_hash}",
            f"--olmo-core-commit-hash {olmo_core_commit_hash}",
            f"--olmo-core-v2-commit-hash {olmo_core_v2_commit_hash}",
            f"--huggingface-transformers-git-url {huggingface_transformers_git_url}",
            f"--huggingface-transformers-commit-hash {huggingface_transformers_commit_hash}",
            (f"--max-sequence-length {max_sequence_length}" if max_sequence_length is not None else ""),
            "--use-system-python",
            ("--skip-validation" if skip_validation else ""),
            (f"--dtype {dtype}" if dtype else ""),
        ]
        remote_command_str = " ".join(remote_command)

        gantry_command = [
            "gantry run",
            f"--description 'Converting OLMo checkpoint at {input_dir}'",
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

    if olmo_type == "olmoe":
        return convert_olmoe(
            input_dir=input_dir,
            unsharded_output_dir=unsharded_output_dir,
            huggingface_output_dir=huggingface_output_dir,
            huggingface_tokenizer=huggingface_tokenizer or DEFAULT_OLMOE_TOKENIZER,
            unsharded_output_suffix=unsharded_output_suffix,
            huggingface_output_suffix=huggingface_output_suffix,
            olmo_commit_hash=olmoe_commit_hash,
            transformers_git_url=huggingface_transformers_git_url,
            transformers_commit_hash=huggingface_transformers_commit_hash,
            env=env,
        )
    elif olmo_type == "olmo2":
        return convert_olmo2(
            input_dir=input_dir,
            unsharded_output_dir=unsharded_output_dir,
            huggingface_output_dir=huggingface_output_dir,
            huggingface_tokenizer=huggingface_tokenizer or DEFAULT_OLMO2_TOKENIZER,
            unsharded_output_suffix=unsharded_output_suffix,
            huggingface_output_suffix=huggingface_output_suffix,
            olmo_commit_hash=olmo2_commit_hash,
            transformers_git_url=huggingface_transformers_git_url,
            transformers_commit_hash=huggingface_transformers_commit_hash,
            env=env,
        )
    elif olmo_type == "olmo-core":
        return convert_olmo_core(
            input_dir=input_dir,
            unsharded_output_dir=unsharded_output_dir,
            huggingface_output_dir=huggingface_output_dir,
            huggingface_tokenizer=huggingface_tokenizer or DEFAULT_OLMO_CORE_TOKENIZER,
            unsharded_output_suffix=unsharded_output_suffix,
            huggingface_output_suffix=huggingface_output_suffix,
            olmo_commit_hash=olmo_core_commit_hash,
            transformers_git_url=huggingface_transformers_git_url,
            transformers_commit_hash=huggingface_transformers_commit_hash,
            env=env,
        )
    elif olmo_type == "olmo-core-v2":
        return convert_olmo_core_v2(
            input_dir=input_dir,
            max_sequence_length=max_sequence_length,
            unsharded_output_dir=unsharded_output_dir,
            huggingface_output_dir=huggingface_output_dir,
            unsharded_output_suffix=unsharded_output_suffix,
            huggingface_output_suffix=huggingface_output_suffix,
            olmo_core_v2_commit_hash=olmo_core_v2_commit_hash,
            transformers_git_url=huggingface_transformers_git_url,
            transformers_commit_hash=huggingface_transformers_commit_hash,
            skip_validation=skip_validation,
            dtype=dtype,
            env=env,
        )
    else:
        raise ValueError(
            f"Invalid olmo type: {olmo_type}; must be one of 'olmo2', 'olmoe', 'olmo-core', or 'olmo-core-v2'"
        )
