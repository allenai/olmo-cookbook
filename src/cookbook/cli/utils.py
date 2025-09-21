import ast
import configparser
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir, mkdtemp
from typing import List, Optional, Union
from urllib.parse import urlparse

from packaging.version import Version

from cookbook.constants import (
    AI2_OLMO_CORE_GIT_URL,
    AI2_OLMO_GIT_URL,
    BEAKER_GANTRY_MAX_VERSION,
    BEAKER_GANTRY_MIN_VERSION,
    BEAKER_PY_MIN_VERSION,
    BEAKER_PY_MAX_VERSION,
    OE_EVAL_GIT_URL,
    TRANSFORMERS_GIT_URL,
    WEKA_MOUNTS,
)


@dataclass(frozen=True)
class PythonEnv:
    name: Optional[str]
    python: str
    pip: str

    @classmethod
    def _find_root_dir(cls, _base: Optional[Path] = None) -> Path:
        if _base is None:
            return cls._find_root_dir(Path(__file__).absolute().parent)

        if (git_dir := (_base / ".git")).exists() and git_dir.is_dir():
            print(f"Found git directory at {_base}; using as venv base location")
            return _base

        if _base.parent == _base:
            print("No git directory found; using system temporary directory as venv location")
            return Path(gettempdir())

        return cls._find_root_dir(_base.parent)

    def path(self) -> dict:
        if self.python == "python":
            # no changes to PATH needed
            return dict(**os.environ)

        path = f"{Path(self.python).parent.absolute()}:{os.environ.get('PATH', '')}"
        return {**os.environ, "PATH": path}

    @classmethod
    def _generate_hash(cls) -> str:
        import hashlib

        hasher = hashlib.sha256()
        hasher.update(cls.python.encode())
        hasher.update(cls.pip.encode())

        return hasher.hexdigest()[:6]

    @classmethod
    def create(cls, name: Optional[str], force: bool = False, root: Optional[Path] = None) -> "PythonEnv":
        name = name or f"oe-py-env-{cls._generate_hash()}"
        path = (root or cls._find_root_dir()) / ".venv" / name

        print(f"Using virtual environment: {path}...")

        if path.exists() and force:
            print(f"Removing existing virtual environment in {path}...")
            shutil.rmtree(path)

        if not path.exists():
            print(f"Creating virtual environment in {path}...")
            subprocess.run(shlex.split(f"python -m venv {path}"), check=True)

        return cls(name=name, python=str(path / "bin" / "python"), pip=str(path / "bin" / "pip"))

    @classmethod
    def null(cls) -> "PythonEnv":
        return cls(name=None, python="python", pip="pip")


def get_huggingface_token() -> Optional[str]:
    if os.getenv("HUGGINGFACE_TOKEN", None):
        return os.getenv("HUGGINGFACE_TOKEN")

    try:
        from huggingface_hub.constants import HF_TOKEN_PATH  # pyright: ignore

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
            pass

    if "AWS_ACCESS_KEY_ID" in os.environ:
        return os.environ["AWS_ACCESS_KEY_ID"]

    if os.path.exists("~/.aws/credentials"):
        with open("~/.aws/credentials", "r") as f:
            for line in f:
                if line.startswith("aws_access_key_id"):
                    return line.split("=")[1].strip()

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
            pass

    if "AWS_SECRET_ACCESS_KEY" in os.environ:
        return os.environ["AWS_SECRET_ACCESS_KEY"]

    if os.path.exists("~/.aws/credentials"):
        with open("~/.aws/credentials", "r") as f:
            for line in f:
                if line.startswith("aws_secret_access_key"):
                    return line.split("=")[1].strip()

    return None


def install_beaker_py(
    env: Optional[PythonEnv] = None,
    beaker_py_max_version: Optional[str] = BEAKER_PY_MAX_VERSION,
    beaker_py_min_version: Optional[str] = BEAKER_PY_MIN_VERSION,
    beaker_gantry_max_version: Optional[str] = BEAKER_GANTRY_MAX_VERSION,
    beaker_gantry_min_version: Optional[str] = BEAKER_GANTRY_MIN_VERSION,
) -> None:
    env = env or PythonEnv.null()

    try:
        return check_beaker_dependencies(
            env=env,
            beaker_py_max_version=beaker_py_max_version,
            beaker_py_min_version=beaker_py_min_version,
            beaker_gantry_min_version=beaker_gantry_min_version,
            beaker_gantry_max_version=beaker_gantry_max_version,
        )
    except Exception:
        cmd = [
            env.pip,
            "install",
            f"beaker-py<={beaker_py_max_version}",
            f"beaker-gantry<={beaker_gantry_max_version}",
        ]

        subprocess.run(shlex.split(" ".join(cmd)), check=True, env=env.path())


def install_oe_eval(
    commit_hash: Optional[str],
    commit_branch: Optional[str],
    env: Optional[PythonEnv] = None,
    no_dependencies: bool = True,
    is_editable: bool = False,
) -> str:
    env = env or PythonEnv.null()

    print("Installing beaker and gantry clients...")
    install_beaker_py(env)

    # Get current installation location, if exists
    result = subprocess.run(
        [env.pip, "show", "oe-eval"],
        capture_output=True,
        text=True
    )

    oe_eval_dir = None
    for line in result.stdout.splitlines():
        if line.startswith("Editable project location:"):
            oe_eval_dir = line.split(":", 1)[1].strip()
            break

    if bool(oe_eval_dir and os.path.exists(oe_eval_dir)):
        # Get local commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=oe_eval_dir,
            capture_output=True,
            text=True
        )
        installed_commit = result.stdout.strip()

        if commit_hash is None:
            # Check if commit matches remote hash (branch or HEAD)
            branch = commit_branch or "HEAD"
            result = subprocess.run(
                ["git", "ls-remote", "origin", branch],
                cwd=oe_eval_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0 or not result.stdout:
                # Fallback to re-clone below
                oe_eval_dir = None
            else:
                remote_commit = result.stdout.split()[0]

                if installed_commit == remote_commit:
                    assert oe_eval_dir is not None
                    print(f"Current commit matches remote {branch} in {oe_eval_dir}")
                    return oe_eval_dir
        else:
            # Check if commit matches user-specified commit
            if installed_commit == commit_hash:
                assert oe_eval_dir is not None
                print(f"Found existing OE-Eval install with matching hash in {oe_eval_dir}")
                return oe_eval_dir

    oe_eval_dir = clone_repository(OE_EVAL_GIT_URL, commit_hash, commit_branch)

    print(f"Installing OE-Eval from {oe_eval_dir}" + (" in editable mode" if is_editable else "") + "...")
    cmd = [
        env.pip,
        "install",
        ("--no-deps" if no_dependencies else ""),
        "--editable" if is_editable else "",
        ".",
    ]
    subprocess.run(shlex.split(" ".join(cmd)), check=True, cwd=oe_eval_dir, env=env.path())

    return oe_eval_dir


def run_func_in_venv(fn):
    """Wrap any function so that it can run inside a virtual environment"""

    def wrapper(*args, env: Optional[PythonEnv] = None, **kwargs):
        if env is None:
            return fn(*args, **kwargs)

        import importlib.util
        import inspect

        package_name = __name__.split(".")[0]
        package_spec = importlib.util.find_spec(package_name)
        assert package_spec is not None and package_spec.origin is not None
        package_path = Path(package_spec.origin).parent
        return_type = inspect.signature(fn).return_annotation

        with NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as f:
            if (fn_module := inspect.getmodule(fn)) is not None:
                f.write(f"from {fn_module.__name__} import {fn.__name__}\n")
            else:
                f.write(inspect.getsource(fn) + "\n")

            f.write(f"out = {fn.__name__}(*{args}, **{kwargs})\n")
            f.write("print(repr(out), end='')\n")
            f.flush()

            result = subprocess.run(
                shlex.split(f"{env.python} {f.name}"),
                capture_output=True,
                # check=True,
                env={**(e := env.path()), "PYTHONPATH": f"{package_path.parent}:{e.get('PYTHONPATH', '')}"},
            )

            if result.returncode != 0:
                print(result.stderr.decode())
                raise RuntimeError("Error running function in virtual environment")

            out = ast.literal_eval(result.stdout.decode())
            assert isinstance(out, return_type), f"Expected {return_type}, got {type(out)}"
            return out

    return wrapper


@run_func_in_venv
def add_secret_to_beaker_workspace(
    secret_name: str,
    secret_value: str,
    workspace: str,
    overwrite: bool = False,
) -> str:
    try:
        import beaker  # pyright: ignore
    except ImportError:
        raise ImportError("beaker-py must be installed to use this function")

    client = beaker.Beaker.from_env(default_workspace=workspace)
    full_secret_name = f"{client.account.name}_{secret_name}"
    try:
        client.secret.get(full_secret_name)
        write_secret = False
    except beaker.exceptions.SecretNotFound:
        write_secret = True

    if write_secret or overwrite:
        client.secret.write(full_secret_name, secret_value)

    return full_secret_name


@run_func_in_venv
def get_beaker_token() -> str:
    try:
        import beaker  # pyright: ignore
    except ImportError:
        raise ImportError("beaker-py must be installed to use this function")

    client = beaker.Beaker.from_env()
    return client.account.config.user_token


@run_func_in_venv
def get_beaker_user() -> str:
    try:
        import beaker  # pyright: ignore
    except ImportError:
        raise ImportError("beaker-py must be installed to use this function")

    client = beaker.Beaker.from_env()
    return client.account.name


@run_func_in_venv
def check_if_secret_exists_in_beaker_workspace(
    secret_name: str,
    workspace: str,
) -> bool:
    try:
        import beaker  # pyright: ignore
    except ImportError:
        raise ImportError("beaker-py must be installed to use this function")

    client = beaker.Beaker.from_env(default_workspace=workspace)
    full_secret_name = f"{client.account.name}_{secret_name}"
    try:
        client.secret.get(full_secret_name)
        return True
    except beaker.exceptions.SecretNotFound:
        return False


def add_aws_flags(
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    workspace: str,
    flags: List[str],
    env: Optional[PythonEnv] = None,
) -> bool:
    if any(flag.startswith("--gantry-secret-aws-access-key-id") for flag in flags) and any(
        flag.startswith("--gantry-secret-aws-secret-access") for flag in flags
    ):
        return True

    if not aws_access_key_id or not aws_secret_access_key:
        # if keys are not set, we are still okay to proceed if they are already set in the workspace
        return check_if_secret_exists_in_beaker_workspace(
            "AWS_ACCESS_KEY_ID", workspace
        ) and check_if_secret_exists_in_beaker_workspace("AWS_SECRET_ACCESS_KEY", workspace)

    aws_access_key_id_secret = add_secret_to_beaker_workspace(
        secret_name="AWS_ACCESS_KEY_ID",
        secret_value=aws_access_key_id,
        workspace=workspace,
        env=env,  # pyright: ignore
    )
    aws_secret_access_key_secret = add_secret_to_beaker_workspace(
        secret_name="AWS_SECRET_ACCESS_KEY",
        secret_value=aws_secret_access_key,
        workspace=workspace,
        env=env,  # pyright: ignore
    )

    flags.append(f"--gantry-secret-aws-access-key-id '{aws_access_key_id_secret}'")
    flags.append(f"--gantry-secret-aws-secret-access '{aws_secret_access_key_secret}'")

    return True


def make_eval_run_name(
    checkpoint_path: str,
    add_bos_token: bool,
    num_shots: int | None = None,
    name_suffix: str | None = None,
) -> str:
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
        + (f"_s{num_shots}" if num_shots is not None else "")
        + (f"_{name_suffix}" if name_suffix else "")
    )


def clone_repository(git_url: str, commit_hash: Optional[str] = None, commit_branch: Optional[str] = None) -> str:
    # current directory
    current_dir = os.getcwd()

    # Prefer HTTPS with token if available to avoid SSH prompts/hangs
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    resolved_url = git_url
    if token and "github.com" in git_url:
        # Convert common SSH forms to HTTPS with token
        if git_url.startswith("git@github.com:"):
            path = git_url.split(":", 1)[1]
            resolved_url = f"https://{token}@github.com/{path}"
        elif git_url.startswith("ssh://git@github.com/"):
            path = git_url.split("github.com/", 1)[1]
            resolved_url = f"https://{token}@github.com/{path}"
        elif git_url.startswith("https://github.com/") and "@github.com" not in git_url:
            path = git_url.split("https://github.com/", 1)[1]
            resolved_url = f"https://{token}@github.com/{path}"

    # Ensure non-interactive git to fail fast if no credentials
    git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_SSH_COMMAND": "ssh -oBatchMode=yes -o StrictHostKeyChecking=accept-new"}

    tmp_dir = None
    try:
        tmp_dir = mkdtemp()

        masked_url = resolved_url.replace(token, "***") if token else resolved_url
        print(f"Cloning repository from {masked_url} to {tmp_dir}...")

        # Base clone command with minimal history
        cmd = shlex.split(f"git clone --depth 1 {resolved_url}")

        if commit_hash:
            cmd.append("--no-checkout")

        cmd.append(tmp_dir)

        # Execute clone
        subprocess.run(cmd, check=True, env=git_env, timeout=300)

        if commit_branch:
            # Change directory to the cloned repo
            os.chdir(tmp_dir)
            subprocess.run(
                shlex.split(f"git fetch origin {commit_branch}:refs/remotes/origin/{commit_branch}"),
                check=True,
                env=git_env,
                timeout=120,
            )
            subprocess.run(
                shlex.split(f"git checkout -b {commit_branch} origin/{commit_branch}"),
                check=True,
                env=git_env,
                timeout=60,
            )

        if commit_hash:
            # Change directory to the cloned repo
            os.chdir(tmp_dir)
            subprocess.run(shlex.split(f"git fetch origin '{commit_hash}'"), check=True, env=git_env, timeout=120)
            subprocess.run(shlex.split(f"git checkout '{commit_hash}'"), check=True, env=git_env, timeout=60)

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


def install_olmo(commit_hash: Optional[str], env: Optional[PythonEnv] = None) -> str:
    env = env or PythonEnv.null()

    # Clone the repository
    olmo_dir = clone_repository(AI2_OLMO_GIT_URL, commit_hash)

    # Install the package
    print(f"Installing OLMo dependencies from {olmo_dir}...")
    subprocess.run(shlex.split(f"{env.pip} install '.[train]'"), check=True, cwd=olmo_dir, env=env.path())

    return olmo_dir


def install_olmo_core(commit_hash: Optional[str], env: Optional[PythonEnv] = None) -> str:
    env = env or PythonEnv.null()

    # Clone the repository
    olmo_dir = clone_repository(AI2_OLMO_CORE_GIT_URL, commit_hash)

    # Removing previous installation
    print("Removing previous installation of ai2-olmo-core...")
    subprocess.run(shlex.split(f"{env.pip} uninstall -y ai2-olmo-core"), cwd=olmo_dir, env=env.path())

    # Install the package
    print(f"Installing OLMo dependencies from {olmo_dir}...")
    subprocess.run(shlex.split(f"{env.pip} install ."), check=True, cwd=olmo_dir, env=env.path())

    return olmo_dir


def make_aws_config(profile_name: str = "default", **kwargs) -> str:
    aws_config = configparser.ConfigParser()
    aws_config[profile_name] = {"region": "us-east-1", "output": "json", **kwargs}

    # Create a StringIO object to serve as a file-like destination
    string_buffer = StringIO()

    # Write the configuration to the StringIO object
    aws_config.write(string_buffer)

    # Get the string value
    return string_buffer.getvalue()


def make_aws_credentials(
    aws_access_key_id: str, aws_secret_access_key: str, profile_name: str = "default", **kwargs
) -> str:
    aws_credentials = configparser.ConfigParser()
    aws_credentials[profile_name] = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        **kwargs,
    }

    string_buffer = StringIO()
    aws_credentials.write(string_buffer)
    return string_buffer.getvalue()


def make_destination_dir(input_dir: str, suffix: str, output_dir: Optional[str] = None) -> str:
    if output_dir is None:
        input_base, input_fn = os.path.split(input_dir)
        output_dir = os.path.join(input_base, f"{input_fn.rstrip('/')}-{suffix}")

    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def download_tokenizer(huggingface_tokenizer: str, env: Optional[PythonEnv] = None) -> str:
    env = env or PythonEnv.null()
    tokenizer_dir = mkdtemp()
    try:
        print(f"Downloading tokenizer from Huggingface Hub to {tokenizer_dir}...")
        cmd = [
            "huggingface-cli",
            "download",
            huggingface_tokenizer,
            f"--local-dir {tokenizer_dir}",
            "--exclude '*safetensors*'",
        ]
        subprocess.run(shlex.split(" ".join(cmd)), check=True, env=env.path())
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        shutil.rmtree(tokenizer_dir, ignore_errors=True)
        raise e

    return tokenizer_dir


def install_transformers(
    commit_hash: Optional[str], env: Optional[PythonEnv] = None, git_url: Optional[str] = None
) -> str:
    env = env or PythonEnv.null()
    git_url = git_url or TRANSFORMERS_GIT_URL

    # Clone the repository
    transformers_dir = clone_repository(git_url, commit_hash)

    # Install the package
    print(f"Installing the correct version of Transformers ({commit_hash}) from {transformers_dir}...")
    subprocess.run(shlex.split(f"{env.pip} install ."), cwd=transformers_dir, check=True, env=env.path())

    print("Installing Huggingface CLI with support for hf_transfer to download tokenizer...")
    subprocess.run(shlex.split(f"{env.pip} install 'huggingface-hub[hf_transfer]'"), check=True, env=env.path())

    print("Installing accelerate to fully support Huggingface model conversion...")
    subprocess.run(shlex.split(f"{env.pip} install 'accelerate>=0.26.0'"), check=True, env=env.path())

    print("Installing torchao to fully support Huggingface model conversion...")
    subprocess.run(shlex.split(f"{env.pip} install 'torchao'"), check=True, env=env.path())

    return transformers_dir


def remove_conflicting_packages(env: Optional[PythonEnv] = None):
    env = env or PythonEnv.null()
    is_flash_attention_installed = (
        subprocess.run(
            shlex.split(f"{env.pip} show flash-attention"), capture_output=True, env=env.path()
        ).returncode
        == 0
    )
    if is_flash_attention_installed:
        print("Uninstalling flash attention to avoid conflicts...")
        subprocess.run(shlex.split(f"{env.pip} uninstall -y flash-attention"), env=env.path())

    is_olmo_core_installed = (
        subprocess.run(
            shlex.split(f"{env.pip} show ai2-olmo-core"), capture_output=True, env=env.path()
        ).returncode
        == 0
    )
    if is_olmo_core_installed:
        print("Uninstalling ai2-olmo-core to avoid conflicts...")
        subprocess.run(shlex.split(f"{env.pip} uninstall -y ai2-olmo-core"), env=env.path())


def check_beaker_dependencies(
    env: Optional[PythonEnv] = None,
    beaker_py_min_version: Optional[str] = BEAKER_PY_MIN_VERSION,
    beaker_py_max_version: Optional[str] = BEAKER_PY_MAX_VERSION,
    beaker_gantry_min_version: Optional[str] = BEAKER_GANTRY_MIN_VERSION,
    beaker_gantry_max_version: Optional[str] = BEAKER_GANTRY_MAX_VERSION,
):
    env = env or PythonEnv.null()
    output = subprocess.run(shlex.split(f"{env.pip} show beaker-py"), capture_output=True, env=env.path())
    if not output.returncode == 0:
        raise RuntimeError("beaker-py must be installed to use this function")
    beaker_py_version_string = next(
        iter([r for r in output.stdout.decode("utf-8").split("\n") if "Version:" in r])
    )
    beaker_py_version = Version(beaker_py_version_string.split(":")[1].strip())

    if beaker_py_max_version is not None and beaker_py_version >= Version(beaker_py_max_version):
        raise RuntimeError(f"beaker-py version v{beaker_py_version} not supported; use <{beaker_py_max_version}")
    if beaker_py_min_version is not None and beaker_py_version < Version(beaker_py_min_version):
        raise RuntimeError(f"beaker-py version v{beaker_py_version} not supported; use >={beaker_py_min_version}")

    gantry_output = subprocess.run(
        shlex.split(f"{env.pip} show beaker-gantry"), capture_output=True, env=env.path()
    )
    if not gantry_output.returncode == 0:
        raise RuntimeError("beaker-gantry must be installed to use this function")
    gantry_version_string = next(
        iter([r for r in gantry_output.stdout.decode("utf-8").split("\n") if "Version:" in r])
    )
    gantry_version = Version(gantry_version_string.split(":")[1].strip())
    if beaker_gantry_max_version is not None and gantry_version >= Version(beaker_gantry_max_version):
        raise RuntimeError(
            f"beaker-gantry v{gantry_version} not supported; use <{beaker_gantry_max_version}"
        )
    if beaker_gantry_min_version is not None and gantry_version < Version(beaker_gantry_min_version):
        raise RuntimeError(f"beaker-gantry v{gantry_version} not supported; use >={beaker_gantry_min_version}")


def find_repository_root(current: Union[str, Path] = __file__) -> Path:
    # go up from current dir until we find a .git directory
    current = Path(current).resolve().absolute()
    if current.is_file():
        return find_repository_root(current.parent)

    if current == Path("/"):
        raise FileNotFoundError("No repository root found")

    if (git_dir := (current / ".git")).exists() and git_dir.is_dir():
        return current

    return find_repository_root(current.parent)


def discover_weka_mount(path: Union[str, Path, None] = None) -> Optional[str]:
    if path is None:
        return None

    _, root, *_ = Path(path).resolve().parts

    if root in WEKA_MOUNTS:
        return root
