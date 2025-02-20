import os
import shlex
import shutil
import subprocess

from cookbook.cli.utils import (
    PythonEnv,
    download_tokenizer,
    install_olmo,
    install_olmo_core,
    install_transformers,
    make_destination_dir,
)
from cookbook.constants import (
    OLMO2_COMMIT_HASH,
    OLMO2_CONVERSION_SCRIPT,
    OLMO2_UNSHARD_SCRIPT,
    OLMO_CORE_COMMIT_HASH,
    OLMO_CORE_UNSHARD_CONVERT_SCRIPT,
    OLMOE_CONVERSION_SCRIPT,
    OLMOE_UNSHARD_SCRIPT,
    TRANSFORMERS_COMMIT_HASH,
    DEFAULT_OLMO2_TOKENIZER,
    DEFAULT_OLMOE_TOKENIZER,
    DEFAULT_OLMO_CORE_TOKENIZER,
)


def convert_olmo_checkpoint(type: str, *args, **kwargs):
    if type == "olmoe":
        convert_olmoe(*args, **kwargs)
    elif type == "olmo2":
        convert_olmo2(*args, **kwargs)
    elif type == "olmo_core":
        convert_olmo_core(*args, **kwargs)
    else:
        raise ValueError(f"Unknown conversion type: {type}")


def convert_olmo_core(
    input_dir: str,
    huggingface_tokenizer: str = DEFAULT_OLMO_CORE_TOKENIZER,
    unsharded_output_dir: str | None = None,
    huggingface_output_dir: str | None = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
    olmo_commit_hash: str = OLMO_CORE_COMMIT_HASH,
    transformers_commit_hash: str = TRANSFORMERS_COMMIT_HASH,
    env: PythonEnv | None = None,
):
    env = env or PythonEnv.null()

    current_directory = os.getcwd()
    directories_to_clean_up = []

    unsharded_output_dir = make_destination_dir(
        input_dir, unsharded_output_suffix, unsharded_output_dir
    )
    huggingface_output_dir = make_destination_dir(
        input_dir, huggingface_output_suffix, huggingface_output_dir
    )

    try:
        print("Starting conversion of OLMo core model...")

        olmo_code_dir = install_olmo_core(env=env, commit_hash=olmo_commit_hash)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(transformers_commit_hash, env)
        directories_to_clean_up.append(huggingface_code_dir)

        tokenizer_dir = download_tokenizer(huggingface_tokenizer, env)
        directories_to_clean_up.append(tokenizer_dir)

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
        print(
            f"Conversion of OLMo core checkpoint complete. Huggingface model saved to {huggingface_output_dir}."
        )

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
    unsharded_output_dir: str | None = None,
    huggingface_output_dir: str | None = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
    olmo_commit_hash: str | None = None,
    transformers_commit_hash: str | None = None,
    env: PythonEnv | None = None,
):
    env = env or PythonEnv.null()

    current_directory = os.getcwd()
    directories_to_clean_up = []

    unsharded_output_dir = make_destination_dir(
        input_dir, unsharded_output_suffix, unsharded_output_dir
    )
    huggingface_output_dir = make_destination_dir(
        input_dir, huggingface_output_suffix, huggingface_output_dir
    )

    try:
        print("Starting conversion of OLMoE model...")

        olmo_code_dir = install_olmo(olmo_commit_hash, env)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(transformers_commit_hash, env)
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
        subprocess.run(
            shlex.split(" ".join(cmd)), check=True, cwd=huggingface_code_dir, env=env.path()
        )
        print(
            f"Conversion of OLMoE checkpoint complete. Huggingface model saved to {huggingface_output_dir}."
        )

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
    unsharded_output_dir: str | None = None,
    huggingface_output_dir: str | None = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
    olmo_commit_hash: str = OLMO2_COMMIT_HASH,
    transformers_commit_hash: str = TRANSFORMERS_COMMIT_HASH,
    env: PythonEnv | None = None,
):
    env = env or PythonEnv.null()
    current_directory = os.getcwd()
    directories_to_clean_up = []

    unsharded_output_dir = make_destination_dir(
        input_dir, unsharded_output_suffix, unsharded_output_dir
    )
    huggingface_output_dir = make_destination_dir(
        input_dir, huggingface_output_suffix, huggingface_output_dir
    )

    try:
        print("Starting conversion of OLMo 2 model...")

        olmo_code_dir = install_olmo(olmo_commit_hash, env)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(transformers_commit_hash, env)
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
        subprocess.run(
            shlex.split(" ".join(cmd)), check=True, cwd=huggingface_code_dir, env=env.path()
        )
        print(
            f"Conversion of OLMo 2 checkpoint complete. Huggingface model saved to {huggingface_output_dir}."
        )

    except Exception as e:
        print(f"Error cloning repositories: {e}")
        os.chdir(current_directory)
        raise e
    finally:
        for directory in directories_to_clean_up:
            print(f"Cleaning up {directory}...")
            shutil.rmtree(directory, ignore_errors=True)
