import os
import shlex
import shutil
import subprocess
import re
from urllib.parse import urlparse
from tempfile import mkdtemp
from typing import Optional, Tuple, List

from cookbook.constants import OE_EVAL_GIT_URL


def get_huggingface_token() -> Optional[str]:
    if os.getenv("HUGGINGFACE_TOKEN", None):
        return os.getenv("HUGGINGFACE_TOKEN")

    try:
        from huggingface_hub.constants import HF_TOKEN_PATH

        with open(HF_TOKEN_PATH, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
    except ImportError:
        return None


def get_aws_access_key_id() -> Optional[str]:
    if shutil.which("aws"):
        try:
            output = subprocess.run(
                shlex.split("aws configure get aws_access_key_id"), capture_output=True, check=True
            )
            return output.stdout.decode().strip()
        except Exception:
            return None
    elif "AWS_ACCESS_KEY_ID" in os.environ:
        return os.environ["AWS_ACCESS_KEY_ID"]
    elif os.path.exists("~/.aws/credentials"):
        with open("~/.aws/credentials", "r") as f:
            for line in f:
                if line.startswith("aws_access_key_id"):
                    return line.split("=")[1].strip()
    else:
        return None


def get_aws_secret_access_key() -> Optional[str]:
    if shutil.which("aws"):
        try:
            output = subprocess.run(
                shlex.split("aws configure get aws_secret_access_key"),
                capture_output=True,
                check=True,
            )
            return output.stdout.decode().strip()
        except Exception:
            return None
    elif "AWS_SECRET_ACCESS_KEY" in os.environ:
        return os.environ["AWS_SECRET_ACCESS_KEY"]
    elif os.path.exists("~/.aws/credentials"):
        with open("~/.aws/credentials", "r") as f:
            for line in f:
                if line.startswith("aws_secret_access_key"):
                    return line.split("=")[1].strip()
    else:
        return None


def clone_repository(git_url: str, commit_hash: Optional[str] = None) -> str:
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


def install_oe_eval(commit_hash: Optional[str] = None) -> str:
    print("Installing beaker and gantry clients...")
    subprocess.run(shlex.split("pip install beaker-py beaker-gantry"), check=True)

    oe_eval_dir = clone_repository(OE_EVAL_GIT_URL, commit_hash)

    print(f"Installing oe-eval toolkit from {oe_eval_dir}...")
    subprocess.run(shlex.split("pip install  --no-deps ."), check=True, cwd=oe_eval_dir)

    return oe_eval_dir


def make_venv(venv_name: str = "oe-eval-venv") -> Tuple[str, str]:
    if not os.path.exists(venv_name):
        subprocess.run(shlex.split(f"python -m venv {venv_name}"), check=True)
    python_exec_abs_path = os.path.abspath(os.path.join(venv_name, "bin", "python"))
    return venv_name, python_exec_abs_path


def destroy_venv(venv_name: str = "oe-eval-venv") -> bool:
    if os.path.exists(venv_name):
        shutil.rmtree(venv_name)
        return True
    return False


def add_secret_to_beaker_workspace(secret_name: str, secret_value: str, workspace: str) -> str:
    import beaker

    client = beaker.Beaker.from_env(default_workspace=workspace)
    full_secret_name = f"{client.account.name}_{secret_name}"
    try:
        client.secret.get(full_secret_name)
    except beaker.exceptions.SecretNotFound:
        client.secret.write(full_secret_name, secret_value)

    return full_secret_name


def add_aws_flags(
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    workspace: str,
    flags: List[str],
) -> bool:
    if any(flag.startswith("--gantry-secret-aws-access-key-id") for flag in flags) and any(
        flag.startswith("--gantry-secret-aws-secret-access") for flag in flags
    ):
        return True

    if not aws_access_key_id or not aws_secret_access_key:
        return False

    aws_access_key_id_secret = add_secret_to_beaker_workspace(
        "AWS_ACCESS_KEY_ID", aws_access_key_id, workspace
    )
    aws_secret_access_key_secret = add_secret_to_beaker_workspace(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key, workspace
    )

    flags.append(f"--gantry-secret-aws-access-key-id '{aws_access_key_id_secret}'")
    flags.append(f"--gantry-secret-aws-secret-access '{aws_secret_access_key_secret}'")

    return True


def make_eval_run_name(checkpoint_path: str, add_bos_token: bool) -> str:
    path_no_scheme = (p := urlparse(checkpoint_path)).netloc + p.path

    step_suffix = None
    if re.search(r"/step\d+[^/]*/?$", checkpoint_path):
        # step name is in the path; we go one level up to get the model name
        step_suffix = os.path.basename(path_no_scheme)
        path_no_scheme = os.path.dirname(path_no_scheme)

    return (
        os.path.basename(path_no_scheme)
        + (f"_{step_suffix}" if step_suffix else "")
        + ("_bos" if add_bos_token else "")
    )
