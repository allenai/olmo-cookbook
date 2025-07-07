#!/usr/bin/env python3
"""
Script to process YAML file and run olmo-cookbook command with latest checkpoint
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


def run_command(cmd, shell=False, errs_okay=False):
    """Run a shell command and return stdout"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"Error: {e.stderr}")
        if not errs_okay:
            sys.exit(1)
        raise e


def get_yaml_name(yaml_file):
    """Extract the 'name' attribute from YAML file"""
    try:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        if "name" not in data:
            print(f"Error: 'name' attribute not found in {yaml_file}")
            sys.exit(1)

        return data["name"]
    except Exception as e:
        print(f"Error reading YAML file {yaml_file}: {e}")
        sys.exit(1)


def get_beaker_name():
    """Get the NAME from 'beaker account whoami' output"""
    output = run_command(["beaker", "account", "whoami"])

    # Parse the table output to extract NAME
    lines = output.strip().split("\n")
    if len(lines) < 2:
        print("Error: Unexpected output from 'beaker account whoami'")
        sys.exit(1)

    # Look for the data row (skip header)
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            return parts[1]  # NAME is the second column

    print("Error: Could not extract NAME from beaker account whoami output")
    sys.exit(1)


def find_latest_checkpoint(beaker_name, yaml_name):
    """Find the latest checkpoint directory in GCS"""
    # Construct the GCS path prefix
    prefix = f"gs://ai2-llm/checkpoints/{beaker_name}/{yaml_name}-"

    # Use gsutil to list directories with the prefix
    cmd = ["gsutil", "ls", "-d", f"{prefix}*/step*/"]

    try:
        output = run_command(cmd)
        if not output:
            print(f"No checkpoints found with prefix: {prefix}")
            sys.exit(1)

        # Get all matching paths
        paths = output.strip().split("\n")

        # Sort paths to get the latest one (lexicographically)
        paths.sort(reverse=True)

        return paths[0].rstrip("/")  # Remove trailing slash

    except subprocess.CalledProcessError as e:
        print(f"Error listing GCS directories: {e.stderr}")
        print(f"Make sure you have access to gs://ai2-llm/checkpoints/{beaker_name}/{yaml_name}-* directories")
        sys.exit(1)


def check_weka_path_exists(gs_path):
    """Check if the corresponding weka path already exists"""
    # Convert gs:// path to weka:// path
    if not gs_path.startswith("gs://"):
        print(f"Error: Expected gs:// path, got: {gs_path}")
        return False

    weka_path = gs_path.replace("gs://", "weka://oe-training-default/")

    # Convert weka:// path to s3:// path for s5cmd
    s3_path = weka_path.replace("weka://oe-training-default/", "s3://oe-training-default/")

    # Add wildcard to check for any files in the directory
    s3_path_wildcard = f"{s3_path}/*"

    print(f"Checking if weka path exists: {weka_path}")
    print(f"Using s5cmd to check: {s3_path_wildcard}")

    cmd = [
        "s5cmd",
        "--profile",
        "WEKA",
        "--endpoint-url",
        "https://weka-aus.beaker.org:9000",
        "ls",
        s3_path_wildcard,
    ]

    try:
        # Run the command - if it succeeds, the path exists
        output = run_command(cmd, errs_okay=True)

        print(f"✅ Weka path exists - found %s files:" % len(output.split("\n")))
        # print(output)
        return True

    except subprocess.CalledProcessError as e:
        # If the command fails, the path doesn't exist
        print(f"❌ Weka path does not exist (s5cmd failed as expected)")
        return False
    except Exception as e:
        print("ERR CODE", e)
        raise e


def run_olmo_cookbook(gs_path):
    """Run the olmo-cookbook command with the GCS path"""
    weka_path = gs_path.replace("gs://", "weka://oe-training-default/")

    cmd = ["python", "-m", "cookbook.remote", gs_path, weka_path]

    print(f"Running: {' '.join(cmd)}")

    try:
        # Run the command and stream output in real-time
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
        )

        beaker_url = None
        beaker_url_pattern = re.compile(r"https://beaker\.org/ex/[A-Z0-9]+")

        for line in process.stdout:
            print(line, end="")

            # Look for the beaker URL in the output
            match = beaker_url_pattern.search(line)
            if match:
                beaker_url = match.group(0)

        process.wait()

        if process.returncode != 0:
            print(f"Error: olmo-cookbook command failed with return code {process.returncode}")
            sys.exit(1)

        # Print the extracted Beaker URL
        if beaker_url:
            print(f"\n" + "=" * 60)
            print(f"🔗 Beaker Experiment URL: {beaker_url}")
            print(f"=" * 60)
            return beaker_url
        else:
            print("\nWarning: Could not extract Beaker experiment URL from output")

    except Exception as e:
        print(f"Error running olmo-cookbook: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Process YAML file and run olmo-cookbook with latest checkpoint")
    parser.add_argument("yaml_file", help="Path to the YAML file")

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.yaml_file).exists():
        print(f"Error: YAML file {args.yaml_file} does not exist")
        sys.exit(1)

    print(f"Processing YAML file: {args.yaml_file}")

    # Step 1: Get name from YAML
    yaml_name = get_yaml_name(args.yaml_file)
    print(f"YAML name: {yaml_name}")

    # Step 2: Get beaker name
    beaker_name = get_beaker_name()
    print(f"Beaker name: {beaker_name}")

    # Step 3: Find latest checkpoint
    print(f"Searching for checkpoints with prefix: gs://ai2-llm/checkpoints/{beaker_name}/{yaml_name}-")
    latest_checkpoint = find_latest_checkpoint(beaker_name, yaml_name)
    print(f"Latest checkpoint: {latest_checkpoint}")

    # Step 4: Check if weka path already exists
    if check_weka_path_exists(latest_checkpoint):
        print(f"\n🚫 Checkpoint already exists in weka storage. Skipping cookbook command.")
        print(f"The checkpoint has already been copied to weka://oe-training-default/")
        return

    # Step 5: Run olmo-cookbook command
    run_olmo_cookbook(latest_checkpoint)


if __name__ == "__main__":
    main()
