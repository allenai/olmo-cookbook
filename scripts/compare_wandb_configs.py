"""

Examples:
    Comparing Peteish7 to OLMoE
    - python scripts/compare_wandb_configs.py https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39 https://wandb.ai/ai2-llm/olmoe/runs/rzsn9tlc

    Comparing Peteish7 to Amberish7
    - python scripts/compare_wandb_configs.py https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39 https://wandb.ai/ai2-llm/olmo-medium/runs/ij4ls6v2


"""

import logging
import os
import re
from collections import Counter

import click
import wandb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from olmo_core.utils import flatten_dict, prepare_cli_environment

log = logging.getLogger(__name__)
run_path_re = re.compile(r"^[^/]+/[^/]+/[^/]+$")
run_path_url = re.compile(r"^https?://wandb.ai/([^/]+)/([^/]+)/runs/([^/]+)")
console = Console()


def parse_run_path(run_path: str) -> str:
    """For convenience, we allow run paths as well as URLs."""
    run_path = run_path.strip("/")
    if run_path_re.match(run_path):
        return run_path

    m = run_path_url.match(run_path)
    if m is not None:
        entity, project, run_id = m.groups()
        return f"{entity}/{project}/{run_id}"

    raise ValueError(f"Could not parse '{run_path}'")


def display_differences_table(left_config, right_config, title):
    # Create exclusive keys tables
    left_only_keys = left_config.keys() - right_config.keys()
    if left_only_keys:
        left_table = Table(title="Settings only in left", title_style="bold cyan")
        left_table.add_column("Key", style="dim")
        left_table.add_column("Value", no_wrap=False)

        for k in sorted(left_only_keys):
            left_table.add_row(str(k), str(left_config[k]))
        console.print(left_table)

    right_only_keys = right_config.keys() - left_config.keys()
    if right_only_keys:
        right_table = Table(title="Settings only in right", title_style="bold magenta")
        right_table.add_column("Key", style="dim")
        right_table.add_column("Value", no_wrap=False)

        for k in sorted(right_only_keys):
            right_table.add_row(str(k), str(right_config[k]))
        console.print(right_table)

    # Create differences table
    keys_with_differences = {
        k for k in left_config.keys() & right_config.keys() if left_config[k] != right_config[k]
    }

    if keys_with_differences:
        diff_table = Table(title=f"Differences in {title}", title_style="bold yellow")
        diff_table.add_column("Parameter", style="dim")
        diff_table.add_column("Left Value", style="cyan")
        diff_table.add_column("Right Value", style="magenta")

        for k in sorted(keys_with_differences):
            diff_table.add_row(str(k), str(left_config[k]), str(right_config[k]))
        console.print(diff_table)
    elif not (left_only_keys or right_only_keys):
        console.print(Panel(f"No differences found in {title}", style="green"))


def display_data_differences(left_data_paths, right_data_paths):
    left_table = Table(title="Data Paths for Left Config", title_style="bold cyan", show_header=True)
    left_table.add_column("Path")
    left_table.add_column("Count", justify="right")

    for path, count in left_data_paths.items():
        left_table.add_row(str(path), str(count))

    right_table = Table(title="Data Paths for Right Config", title_style="bold magenta", show_header=True)
    right_table.add_column("Path")
    right_table.add_column("Count", justify="right")

    for path, count in right_data_paths.items():
        right_table.add_row(str(path), str(count))

    console.print(left_table)
    console.print(right_table)


@click.command()
@click.argument(
    "left_run_path",
    type=str,
)
@click.argument(
    "right_run_path",
    type=str,
)
@click.option(
    "--diff-datasets",
    is_flag=True,
    default=False,
    help="Whether to compare dataset differences between runs",
)
def main(
    left_run_path: str,
    right_run_path: str,
    diff_datasets: bool,
):
    api = wandb.Api()
    left_run = api.run(parse_run_path(left_run_path))
    right_run = api.run(parse_run_path(right_run_path))

    left_config_raw = left_run._attrs["rawconfig"]
    right_config_raw = right_run._attrs["rawconfig"]

    # flattening the dict will make diffs easier
    left_config = flatten_dict(left_config_raw)
    right_config = flatten_dict(right_config_raw)

    # Handle dataset paths conditionally based on diff_datasets flag
    left_data_paths = Counter()
    right_data_paths = Counter()
    if diff_datasets and "dataset.paths" in left_config:
        left_data_paths = Counter([os.path.dirname(path) for path in left_config["dataset.paths"]])
        del left_config["dataset.paths"]
    elif "dataset.paths" in left_config:
        del left_config["dataset.paths"]

    if diff_datasets and "dataset.paths" in right_config:
        right_data_paths = Counter([os.path.dirname(path) for path in right_config["dataset.paths"]])
        del right_config["dataset.paths"]
    elif "dataset.paths" in right_config:
        del right_config["dataset.paths"]

    # Handle source_mixture_config in the same way
    if "dataset.source_mixture_config.source_configs" in left_config:
        source_configs = left_config["dataset.source_mixture_config.source_configs"]
        if diff_datasets:
            for config in source_configs:
                if isinstance(config, dict) and "paths" in config:
                    paths = config["paths"]
                    for path in paths:
                        left_data_paths[os.path.dirname(path)] += 1

        for config in source_configs:
            if isinstance(config, dict) and "paths" in config:
                del config["paths"]

        left_config["dataset.source_mixture_config.source_configs"] = source_configs

    if "dataset.source_mixture_config.source_configs" in right_config:
        source_configs = right_config["dataset.source_mixture_config.source_configs"]
        if diff_datasets:
            for config in source_configs:
                if isinstance(config, dict) and "paths" in config:
                    paths = config["paths"]
                    for path in paths:
                        right_data_paths[os.path.dirname(path)] += 1

        for config in source_configs:
            if isinstance(config, dict) and "paths" in config:
                del config["paths"]

        del right_config["dataset.source_mixture_config.source_configs"]

    # Display header with run information
    console.print()
    console.rule(f"[bold]Config differences between runs[/bold]")
    console.print(f"Left:  [cyan]{left_run_path}[/cyan]")
    console.print(f"Right: [magenta]{right_run_path}[/magenta]")
    console.print()

    # Display parameter differences
    console.rule("[bold]Parameter Differences[/bold]")
    display_differences_table(left_config, right_config, "parameters")
    console.print()

    # Display data differences only if diff_datasets is enabled
    if diff_datasets:
        console.rule("[bold]Data Differences[/bold]")
        display_data_differences(left_data_paths, right_data_paths)
        console.print()


if __name__ == "__main__":
    prepare_cli_environment()
    main()
