import os
import shlex
import shutil
import subprocess
from pathlib import Path
from tempfile import mkdtemp

from cookbook.constants import (
    AI2_OLMO_CORE_GIT_URL,
    AI2_OLMO_GIT_URL,
    # BEAKER_DEFAULT_BUDGET,
    # BEAKER_DEFAULT_PRIORITY,
    # BEAKER_DEFAULT_WORKSPACE,
    BEAKER_KNOWN_CLUSTERS,
    DEFAULT_OLMOE_TOKENIZER,
    DEFAULT_OLMO2_TOKENIZER,
    DEFAULT_OLMO_CORE_TOKENIZER,
    OLMOE_COMMIT_HASH,
    OLMOE_UNSHARD_SCRIPT,
    OLMO2_COMMIT_HASH,
    OLMO2_UNSHARD_SCRIPT,
    OLMO2_CONVERSION_SCRIPT,
    OLMOE_CONVERSION_SCRIPT,
    OLMO_CORE_COMMIT_HASH,
    OLMO_CORE_UNSHARD_CONVERT_SCRIPT,
    OLMO_TYPES,
    TRANSFORMERS_COMMIT_HASH,
    TRANSFORMERS_GIT_URL,
    WEKA_MOUNTS,
)


def clone_repository(git_url: str, commit_hash: str | None = None) -> str:
    # current directory
    current_dir = os.getcwd()

    tmp_dir = None
    try:
        tmp_dir = mkdtemp()

        # Base clone command with minimal history
        cmd = shlex.split(f"git clone --depth 1 {git_url}")

        if commit_hash:
            cmd.append("--no-checkout")

        cmd.append(tmp_dir)

        # Execute clone
        subprocess.run(cmd, check=True)

        if commit_hash:
            # Change directory to the cloned repo
            os.chdir(tmp_dir)
            subprocess.run(shlex.split(f"git fetch origin '{commit_hash}'"), check=True)
            subprocess.run(shlex.split(f"git checkout  '{commit_hash}'"), check=True)

        return tmp_dir

    except Exception as e:
        print(
            f"Error cloning repository at '{git_url}' {f' (commit {commit_hash})' if commit_hash else ''}: {e}"
        )
        if tmp_dir:
            print(f"Cleaning up {tmp_dir}...")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise e
    finally:
        os.chdir(current_dir)


def install_olmo(commit_hash: str | None) -> str:
    # Clone the repository
    olmo_dir = clone_repository(AI2_OLMO_GIT_URL, commit_hash)

    # Install the package
    print(f"Installing OLMo dependencies from {olmo_dir}...")
    subprocess.run(shlex.split("pip install '.[train]'"), check=True, cwd=olmo_dir)

    return olmo_dir


def install_olmo_core(commit_hash: str | None) -> str:
    # Clone the repository
    olmo_dir = clone_repository(AI2_OLMO_CORE_GIT_URL, commit_hash)

    # Install the package
    print(f"Installing OLMo dependencies from {olmo_dir}...")
    subprocess.run(shlex.split("pip install ."), check=True, cwd=olmo_dir)

    return olmo_dir


def install_transformers(commit_hash: str | None) -> str:
    # Clone the repository
    transformers_dir = clone_repository(TRANSFORMERS_GIT_URL, commit_hash)

    # Install the package
    print(
        f"Installing the correct version of Transformers ({commit_hash}) from {transformers_dir}..."
    )
    subprocess.run(shlex.split("pip install ."), check=True, cwd=transformers_dir)

    print("Installing Huggingface CLI with support for hf_transfer to download tokenizer...")
    subprocess.run(shlex.split("pip install 'huggingface-hub[hf_transfer]'"), check=True)

    print("Installing accelerate to fully support Huggingface model conversion...")
    subprocess.run(shlex.split("pip install 'accelerate>=0.26.0'"), check=True)

    return transformers_dir


def make_destination_dir(input_dir: str, suffix: str, output_dir: str | None = None) -> str:
    if output_dir is None:
        input_base, input_fn = os.path.split(input_dir)
        output_dir = os.path.join(input_base, f"{input_fn.rstrip('/')}-{suffix}")

    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def download_tokenizer(huggingface_tokenizer: str) -> str:
    tokenizer_dir = None
    try:
        tokenizer_dir = mkdtemp()
        print(f"Downloading tokenizer from Huggingface Hub to {tokenizer_dir}...")
        subprocess.run(
            shlex.split(
                f"huggingface-cli download {huggingface_tokenizer} --local-dir {tokenizer_dir}"
            ),
            check=True,
        )
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        if tokenizer_dir:
            print(f"Cleaning up {tokenizer_dir}...")
            shutil.rmtree(tokenizer_dir, ignore_errors=True)
        raise e

    return tokenizer_dir


def convert_olmoe(
    input_dir: str,
    huggingface_tokenizer: str,
    unsharded_output_dir: str | None = None,
    huggingface_output_dir: str | None = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
):
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

        olmo_code_dir = install_olmo(OLMOE_COMMIT_HASH)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(TRANSFORMERS_COMMIT_HASH)
        directories_to_clean_up.append(huggingface_code_dir)

        model_dir = os.path.join(input_dir, "model")
        is_model_in_directory = os.path.exists(model_dir) and os.path.isdir(model_dir)

        tokenizer_dir = download_tokenizer(huggingface_tokenizer)
        directories_to_clean_up.append(tokenizer_dir)

        if is_model_in_directory:
            print("OLMo checkpoint is sharded. Unsharding...")
            unshard_command = (
                f"python {OLMOE_UNSHARD_SCRIPT} '{input_dir}' '{unsharded_output_dir}' --model-only"
            )
            os.makedirs(unsharded_output_dir, exist_ok=True)
            subprocess.run(shlex.split(unshard_command), check=True, cwd=olmo_code_dir)
            print(f"Unsharding to {unsharded_output_dir} complete.")
        else:
            print("OLMo checkpoint is not sharded. Skipping unsharding...")
            shutil.rmtree(unsharded_output_dir, ignore_errors=True)
            unsharded_output_dir = input_dir

        print("Converting OLMoE weights to Huggingface format...")
        os.makedirs(huggingface_output_dir, exist_ok=True)
        cmd = [
            "python",
            OLMOE_CONVERSION_SCRIPT,
            "--input_dir",
            unsharded_output_dir,
            "--tokenizer_json_path",
            os.path.join(tokenizer_dir, "tokenizer.json"),
            "--output_dir",
            huggingface_output_dir,
        ]
        subprocess.run(cmd, check=True, cwd=huggingface_code_dir)
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
    huggingface_tokenizer: str,
    unsharded_output_dir: str | None = None,
    huggingface_output_dir: str | None = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
):
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

        olmo_code_dir = install_olmo(OLMO2_COMMIT_HASH)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(TRANSFORMERS_COMMIT_HASH)
        directories_to_clean_up.append(huggingface_code_dir)

        model_dir = os.path.join(input_dir, "model")
        is_model_in_directory = os.path.exists(model_dir) and os.path.isdir(model_dir)

        tokenizer_dir = download_tokenizer(huggingface_tokenizer)
        directories_to_clean_up.append(tokenizer_dir)

        if is_model_in_directory:
            print("OLMo checkpoint is sharded. Unsharding...")
            unshard_command = (
                f"python {OLMO2_UNSHARD_SCRIPT} '{input_dir}' '{unsharded_output_dir}' --model-only"
            )
            os.makedirs(unsharded_output_dir, exist_ok=True)
            subprocess.run(shlex.split(unshard_command), check=True, cwd=olmo_code_dir)
            print(f"Unsharding to {unsharded_output_dir} complete.")
        else:
            print("OLMo checkpoint is not sharded. Skipping unsharding...")
            shutil.rmtree(unsharded_output_dir, ignore_errors=True)
            unsharded_output_dir = input_dir

        print("Converting OLMo 2 weights to Huggingface format...")
        os.makedirs(huggingface_output_dir, exist_ok=True)
        cmd = [
            "python",
            OLMO2_CONVERSION_SCRIPT,
            "--input_dir",
            unsharded_output_dir,
            "--tokenizer_json_path",
            os.path.join(tokenizer_dir, "tokenizer.json"),
            "--output_dir",
            huggingface_output_dir,
        ]
        subprocess.run(shlex.split(" ".join(cmd)), check=True, cwd=huggingface_code_dir)
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


def remove_conflicting_packages():
    is_flash_attention_installed = (
        subprocess.run(shlex.split("pip show flash-attention"), capture_output=True).returncode == 0
    )
    if is_flash_attention_installed:
        print("Uninstalling flash attention to avoid conflicts...")
        subprocess.run(shlex.split("pip uninstall -y flash-attention"))


def check_beaker_dependencies():
    is_beaker_py_installed = (
        subprocess.run(shlex.split("pip show beaker-py"), capture_output=True).returncode == 0
    )
    is_beaker_gantry_installed = (
        subprocess.run(shlex.split("pip show beaker-gantry"), capture_output=True).returncode == 0
    )

    if not is_beaker_py_installed or not is_beaker_gantry_installed:
        raise RuntimeError(
            "When using --beaker, both beaker-py and beaker-gantry must be installed"
        )


def discover_weka_mount(path: str | Path | None = None) -> str | None:
    if path is None:
        return None

    _, root, *_ = Path(path).resolve().parts

    if root in WEKA_MOUNTS:
        return root


def add_secret_to_beaker_workspace(secret_name: str, secret_value: str, workspace: str):
    try:
        import beaker  # pyright: ignore
    except ImportError:
        return None

    client = beaker.Beaker.from_env(default_workspace=workspace)
    full_secret_name = f"{client.account.name}_{secret_name}"
    try:
        client.secret.get(full_secret_name)
    except beaker.exceptions.SecretNotFound:
        client.secret.write(full_secret_name, secret_value)

    return full_secret_name


def find_repository_root(current: str | Path = __file__) -> Path:
    # go up from current dir until we find a .git directory
    current = Path(current).resolve().absolute()
    if current.is_file():
        return find_repository_root(current.parent)

    if current == Path("/"):
        raise FileNotFoundError("No repository root found")

    if (git_dir := (current / ".git")).exists() and git_dir.is_dir():
        return current

    return find_repository_root(current.parent)


def convert_olmo_core(
    input_dir: str,
    huggingface_tokenizer: str,
    unsharded_output_dir: str | None = None,
    huggingface_output_dir: str | None = None,
    unsharded_output_suffix: str = "unsharded",
    huggingface_output_suffix: str = "hf",
):
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

        olmo_code_dir = install_olmo_core(OLMO_CORE_COMMIT_HASH)
        directories_to_clean_up.append(olmo_code_dir)

        huggingface_code_dir = install_transformers(TRANSFORMERS_COMMIT_HASH)
        directories_to_clean_up.append(huggingface_code_dir)

        tokenizer_dir = download_tokenizer(huggingface_tokenizer)
        directories_to_clean_up.append(tokenizer_dir)

        print("Converting OLMo core weights to Huggingface format...")
        os.makedirs(huggingface_output_dir, exist_ok=True)
        cmd = [
            "python",
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
        subprocess.run(shlex.split(" ".join(cmd)), check=True, cwd=olmo_code_dir)
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


def convert_checkpoint(
    input_dir: str,
    olmo_type: str,
    huggingface_tokenizer: str | None,
    unsharded_output_dir: str | None,
    huggingface_output_dir: str | None,
    unsharded_output_suffix: str,
    huggingface_output_suffix: str,
    olmoe_commit_hash: str,
    olmo2_commit_hash: str,
    huggingface_token: str | None,
    use_beaker: bool,
    beaker_workspace: str,
    beaker_priority: str,
    beaker_cluster: str,
    beaker_allow_dirty: bool,
    beaker_budget: str,
    beaker_gpus: int,
    beaker_dry_run: bool,
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

    if olmo_type == "olmoe":
        convert_olmoe(
            input_dir=input_dir,
            unsharded_output_dir=unsharded_output_dir,
            huggingface_output_dir=huggingface_output_dir,
            huggingface_tokenizer=huggingface_tokenizer or DEFAULT_OLMOE_TOKENIZER,
            unsharded_output_suffix=unsharded_output_suffix,
            huggingface_output_suffix=huggingface_output_suffix,
        )
    elif olmo_type == "olmo2":
        convert_olmo2(
            input_dir=input_dir,
            unsharded_output_dir=unsharded_output_dir,
            huggingface_output_dir=huggingface_output_dir,
            huggingface_tokenizer=huggingface_tokenizer or DEFAULT_OLMO2_TOKENIZER,
            unsharded_output_suffix=unsharded_output_suffix,
            huggingface_output_suffix=huggingface_output_suffix,
        )
    elif olmo_type == "olmo-core":
        convert_olmo_core(
            input_dir=input_dir,
            unsharded_output_dir=unsharded_output_dir,
            huggingface_output_dir=huggingface_output_dir,
            huggingface_tokenizer=huggingface_tokenizer or DEFAULT_OLMO_CORE_TOKENIZER,
            unsharded_output_suffix=unsharded_output_suffix,
            huggingface_output_suffix=huggingface_output_suffix,
        )
    else:
        raise ValueError(f"Invalid olmo type: {olmo_type}; must be one of {OLMO_TYPES}")
