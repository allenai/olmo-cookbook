#!/usr/bin/env python3
"""Find failed experiments in ai2/dolma2 workspace with specific prefix or by experiment ID."""

import json
import argparse
import logging
from beaker import Beaker
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

default_prefix = "lmeval-5xC-30m-superswarm-dclm-stackedu-conditional"
default_workspace = "ai2/dolma2"


# Convert all datetime objects to strings for JSON serialization
def convert_datetimes(obj):
    if isinstance(obj, dict):
        return {k: convert_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_datetimes(item) for item in obj]
    elif hasattr(obj, "isoformat"):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):  # custom objects with attributes
        return str(obj)
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Find failed experiments by prefix or specific experiment ID")
    parser.add_argument("--experiment-id", "-i", help="Specific experiment ID to check")
    parser.add_argument("--prefix", "-p", default=default_prefix, help="Prefix to filter experiments")
    parser.add_argument("--workspace", "-w", default=default_workspace, help="Beaker workspace")

    args = parser.parse_args()

    console = Console()
    client = Beaker.from_env()

    failed_experiments = []

    if args.experiment_id:
        # Fast path: check specific experiment by ID
        console.print(f"[bold blue]Checking experiment: {args.experiment_id}[/bold blue]")

        try:
            exp_details = client.experiment.get(args.experiment_id)
            json_obj = exp_details.model_dump()
            json_obj = convert_datetimes(json_obj)
            console.print_json(data=json_obj, indent=2)
            if exp_details.jobs[0].status.exit_code != 0:
                failed_experiments.append(
                    {
                        "id": exp_details.id,
                        "name": exp_details.name,
                    }
                )
        except Exception as e:
            console.print(f"[bold red]Error fetching experiment {args.experiment_id}: {e}[/bold red]")
            return

    else:
        # Original path: search by prefix
        console.print(f"[bold blue]Fetching experiments from workspace: {args.workspace}[/bold blue]")

        # Get all experiments in the workspace
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Loading experiments...", total=None)
            experiments = client.workspace.experiments(args.workspace)
            progress.update(task, completed=True)

        # Filter experiments by prefix
        matching_experiments = [exp for exp in experiments if exp.name and exp.name.startswith(args.prefix)]

        console.print(f"Found {len(matching_experiments)} experiments with prefix '{args.prefix}'")

        # Check status of each matching experiment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Checking experiment status...", total=len(matching_experiments))

            for exp in matching_experiments:
                # Get experiment details to check status
                exp_details = client.experiment.get(exp.id)
                json_obj = exp_details.model_dump()
                json_obj = convert_datetimes(json_obj)

                if exp_details.jobs[0].status.exit_code != 0:
                    failed_experiments.append(
                        {
                            "id": exp_details.id,
                            "name": exp_details.name,
                        }
                    )
                progress.advance(task)

    # Output to JSON file
    output_file = f"{args.experiment_id or args.prefix}_failed.json"
    with open(output_file, "w") as f:
        json.dump(failed_experiments, f, indent=2)

    console.print(f"[bold green]Found {len(failed_experiments)} failed experiments[/bold green]")
    console.print(f"[bold green]Results saved to {output_file}[/bold green]")


if __name__ == "__main__":
    main()
