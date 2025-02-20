import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir, mkdtemp
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from cookbook.constants import (
    AI2_OLMO_CORE_GIT_URL,
    AI2_OLMO_GIT_URL,
    OE_EVAL_GIT_URL,
    TRANSFORMERS_GIT_URL,
    WEKA_MOUNTS,
)


@dataclass(frozen=True)
class PythonEnv:
    name: str | None
    python: str
    pip: str

    @classmethod
    def _find_root_dir(cls, _base: Path | None = None) -> Path:
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
    def create(
        cls, name: Optional[str], force: bool = False, root: Path | None = None
    ) -> "PythonEnv":
        name = name or f"oe-py-env-{cls._generate_hash()}"
        path = (root or cls._find_root_dir()) / ".venv" / name
        if path.exists() and force:
            shutil.rmtree(path)

        if not path.exists():
            subprocess.run(shlex.split(f"python -m venv {path}"), check=True)

        return cls(name=name, python=str(path / "bin" / "python"), pip=str(path / "bin" / "pip"))

    @classmethod
    def null(cls) -> "PythonEnv":
        return cls(name=None, python="python", pip="pip")


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


def install_oe_eval(env: PythonEnv = PythonEnv.null(), commit_hash: Optional[str] = None) -> str:
    print("Installing beaker and gantry clients...")
    subprocess.run(shlex.split("pip install beaker-py beaker-gantry"), check=True, env=env.path())

    oe_eval_dir = clone_repository(OE_EVAL_GIT_URL, commit_hash)

    print(f"Installing oe-eval toolkit from {oe_eval_dir}...")
    subprocess.run(shlex.split("pip install  --no-deps ."), check=True, cwd=oe_eval_dir)

    return oe_eval_dir


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


def install_olmo(commit_hash: str | None, env: PythonEnv | None = None) -> str:
    env = env or PythonEnv.null()

    # Clone the repository
    olmo_dir = clone_repository(AI2_OLMO_GIT_URL, commit_hash)

    # Install the package
    print(f"Installing OLMo dependencies from {olmo_dir}...")
    subprocess.run(
        shlex.split(f"{env.pip} install '.[train]'"), check=True, cwd=olmo_dir, env=env.path()
    )

    return olmo_dir


def install_olmo_core(commit_hash: str | None, env: PythonEnv | None = None) -> str:
    env = env or PythonEnv.null()

    # Clone the repository
    olmo_dir = clone_repository(AI2_OLMO_CORE_GIT_URL, commit_hash)

    # Install the package
    print(f"Installing OLMo dependencies from {olmo_dir}...")
    subprocess.run(shlex.split(f"{env.pip} install ."), check=True, cwd=olmo_dir, env=env.path())

    return olmo_dir


def make_destination_dir(input_dir: str, suffix: str, output_dir: str | None = None) -> str:
    if output_dir is None:
        input_base, input_fn = os.path.split(input_dir)
        output_dir = os.path.join(input_base, f"{input_fn.rstrip('/')}-{suffix}")

    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def download_tokenizer(huggingface_tokenizer: str, env: PythonEnv | None = None) -> str:
    env = env or PythonEnv.null()
    tokenizer_dir = mkdtemp()
    try:
        print(f"Downloading tokenizer from Huggingface Hub to {tokenizer_dir}...")
        cmd = f"huggingface-cli download {huggingface_tokenizer} --local-dir {tokenizer_dir}"
        subprocess.run(shlex.split(cmd), check=True, env=env.path())
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        shutil.rmtree(tokenizer_dir, ignore_errors=True)
        raise e

    return tokenizer_dir


def install_transformers(commit_hash: str | None, env: PythonEnv | None = None) -> str:
    env = env or PythonEnv.null()

    # Clone the repository
    transformers_dir = clone_repository(TRANSFORMERS_GIT_URL, commit_hash)

    # Install the package
    print(
        f"Installing the correct version of Transformers ({commit_hash}) from {transformers_dir}..."
    )
    subprocess.run(
        shlex.split(f"{env.pip} install ."), cwd=transformers_dir, check=True, env=env.path()
    )

    print("Installing Huggingface CLI with support for hf_transfer to download tokenizer...")
    subprocess.run(
        shlex.split(f"{env.pip} install 'huggingface-hub[hf_transfer]'"), check=True, env=env.path()
    )

    print("Installing accelerate to fully support Huggingface model conversion...")
    subprocess.run(
        shlex.split(f"{env.pip} install 'accelerate>=0.26.0'"), check=True, env=env.path()
    )

    return transformers_dir


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


def discover_weka_mount(path: str | Path | None = None) -> str | None:
    if path is None:
        return None

    _, root, *_ = Path(path).resolve().parts

    if root in WEKA_MOUNTS:
        return root
