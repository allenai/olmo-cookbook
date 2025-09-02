#!/usr/bin/env python3

import argparse
import sys
import json
import subprocess
import shutil


TASKS = [
    "gen::xlarge",
    "mbpp:3shot::olmo3:n32:v2",
    "minerva",
    "olmo3:dev:7b:mcqa:stem",
    "olmo3:dev:7b:mcqa:non_stem"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model name or path")
    parser.add_argument("--dashboard", type=str, default="peteish-LC-ruler")
    parser.add_argument("--tasks", type=str, default=TASKS, nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model
    dashboard_name = args.dashboard

    if 'ai2-llm/checkpoints' in model_name:
        # make model name from model_path
        *_, model_name, step = model_name.rsplit('/')
        if not step.endswith('-hf'):
            step = step + '-hf'
        model_name = f"{model_name}_{step}"

    # Check if olmo-cookbook-eval command is available
    eval_command = None
    if shutil.which("olmo-cookbook-eval"):
        eval_command = "olmo-cookbook-eval"
    elif shutil.which("uv"):
        # Check if uv run olmo-cookbook-eval works
        try:
            subprocess.run(["uv", "run", "olmo-cookbook-eval", "--help"],
                         capture_output=True, check=True)
            eval_command = "uv run olmo-cookbook-eval"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not eval_command:
        print("Error: olmo-cookbook-eval command not found. Please install it or ensure uv is available.")
        sys.exit(1)

    tasks_regex = "^(" + "|".join(args.tasks) + ")$"

    print(f"Grabbing {tasks_regex}", file=sys.stderr)

    # Build the command
    cmd = f"{eval_command} results --dashboard {dashboard_name} --tasks '{tasks_regex}' -m '{model_name}' --format json"

    # Execute the command and capture output
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        csv_out: list[str] = [""] * (len(args.tasks) + 3)

        for task_name, values in data.items():
            if task_name not in args.tasks:
                continue

            task_length = 4096

            assert len(values) == 1, "more than one matching model"
            model_name_from_data = list(values.keys())[0]
            task_value = values[model_name_from_data]

            try:
                task_value = str(float(task_value) * 100)
            except TypeError:
                task_value = "-"

            csv_out[0] = str(task_length)
            csv_out[1] = model_name_from_data
            csv_out[args.tasks.index(task_name) + 3] = task_value

        print(",".join(csv_out))

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
