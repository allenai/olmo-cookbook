#!/usr/bin/env python3

import sys
import json
import subprocess
import shutil
import re

def main():
    if len(sys.argv) < 2:
        print("Usage: python grab_ruler.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    dashboard_name = "peteish-LC-ruler"

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

    ruler_sets = ["ruler:4k", "ruler:8k", "ruler:16k", "ruler:32k", "ruler:64k"]

    for ruler_set in ruler_sets:
        print(f"Grabbing {ruler_set}", file=sys.stderr)

        # Build the command
        cmd = f"{eval_command} results --dashboard {dashboard_name} --tasks '^{ruler_set}$' -m '{model_name}' --format json"

        # Execute the command and capture output
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Process the data
            order = [
                "niah_s_1",
                "niah_s_2",
                "niah_s_3",
                "niah_mk_1",
                "niah_mk_2",
                "niah_mk_3",
                "niah_mv",
                "niah_mq",
                "vt",
                "cwe",
                "fwe",
                "qa_1",
                "qa_2"
            ]

            csv_out = [""] * (len(order) + 3)

            for task, values in data.items():
                # ruler_qa_1__16384::std
                task_match = re.match(r"ruler_(?P<task_name>[a-z0-9_]+)__(?P<task_length>\d+)::std", task)

                if not task_match:
                    continue

                task_name = task_match.group("task_name")
                task_length = task_match.group("task_length")

                if task_name not in order:
                    continue

                task_length = task.split("__")[1].split("::")[0]

                assert len(values) == 1, "more than one matching model"
                model_name_from_data = list(values.keys())[0]
                task_value = values[model_name_from_data]

                csv_out[0] = task_length
                csv_out[1] = model_name_from_data
                csv_out[order.index(task_name) + 3] = str(float(task_value) * 100)

            print(",".join(csv_out))

        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
