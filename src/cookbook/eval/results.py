from cookbook.cli.utils import (
    PythonEnv,
    add_aws_flags,
    add_secret_to_beaker_workspace,
    install_oe_eval,
    make_eval_run_name,
)



def get_results(
    checkpoint_path: list[str],
    add_bos_token: bool,
    dashboard: str,
    tasks: list[str],
    dry_run: bool,
    python_venv_force: bool,
    python_venv_name: str,
):
    env = PythonEnv.create(name=python_venv_name, force=python_venv_force)
    print(f"Using Python virtual environment at {env.name}")
