import shlex
import subprocess
from dataclasses import InitVar, dataclass

from cookbook.cli.utils import (
    PythonEnv,
    add_secret_to_beaker_workspace,
    install_beaker_py,
)
from cookbook.utils.clusters import get_matching_clusters

from .base import LocatedPath


@dataclass
class GantryLauncher:
    allow_dirty: bool
    budget: str
    cluster: str
    dry_run: bool
    gpus: int
    priority: str
    preemptible: bool
    workspace: str
    env: InitVar[PythonEnv | None] = None

    def __post_init__(self, env: PythonEnv | None):
        self._env = env or PythonEnv.null()
        self._flags = []

        # setup beaker-py
        install_beaker_py(env=self._env)

        for cluster in set(get_matching_clusters(self.cluster)):
            self._flags.append(f"--cluster {cluster}")

    def add_mount(self, path: str):
        located_path = LocatedPath.from_str(path)
        if located_path.prot == "weka":
            self._flags.append(f"--weka {located_path.bucket}:/{located_path.bucket}")

    def add_env_secret(self, key: str, value: str, overwrite: bool = False):
        secret_name = add_secret_to_beaker_workspace(
            secret_name=key,
            secret_value=value,
            workspace=self.workspace,
            env=self._env,  # pyright: ignore
            overwrite=overwrite,
        )
        self._flags.append(f"--env-secret {key}={secret_name}")

    def run(
        self,
        command: str,
        description: str,
        extra_flags: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:

        extra_flags = extra_flags or {}
        breakpoint()

        gantry_command = [
            "gantry run",
            f"--description '{description}'",
            ("--allow-dirty" if self.allow_dirty else ""),
            "--no-python",
            "--uv-venv /stage/.venv",
            f"--workspace {self.workspace}",
            f"--priority {self.priority}",
            f"--gpus {self.gpus}",
            ("--preemptible" if self.preemptible else ""),
            f"--budget {self.budget}",
            "--yes",
            ("--dry-run" if self.dry_run else ""),
            " ".join(self._flags),
            " ".join(f"--{k} {v}" for k, v in extra_flags.items()),
            # f"-- /bin/bash -c 'pip install uv && uv pip install . --system && {command}'",
            f"-- /bin/bash -c '{command}'",
        ]
        gantry_command_str = " ".join(gantry_command)

        print(f"Submitting to beaker with command: {gantry_command_str}")
        return subprocess.run(shlex.split(gantry_command_str), check=True, env=self._env.path())
