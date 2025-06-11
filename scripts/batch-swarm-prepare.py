#!/usr/bin/env python3

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

checkpoint_base_path = "ai2-llm/checkpoints/"


def mk_move_ckpt(base_path: str, index: str):
    """Generate the command to move a checkpoint."""
    return [
        "python",
        "-m",
        "cookbook.remote",
        f"gs://{checkpoint_base_path}{base_path}-{index}",
        f"weka://oe-data-default/{checkpoint_base_path}{base_path}-{index}",
        "--allow-dirty",
        "--workspace",
        "ai2/dolma2",
    ]


def mk_convert_ckpt(base_path: str, index: str):
    """Generate the command to convert a checkpoint."""
    return [
        "uv",
        "run",
        "olmo-cookbook-eval",
        "convert",
        f"/oe-data-default/{checkpoint_base_path}{base_path}-{index}/step22100",
        "-t",
        "olmo-core-v2",
        "--beaker-allow-dirty",
        "--use-beaker",
        "--beaker-priority",
        "urgent",
    ]


def mk_eval_ckpt(base_path: str, index: str):
    """Generate the command to evaluate a checkpoint."""
    checkpoint_path = f"/oe-data-default/{checkpoint_base_path}{base_path}-{index}/step22100-hf"

    # VLLM evaluation command
    vllm_cmd = [
        "uv",
        "run",
        "olmo-cookbook-eval",
        "evaluate",
        checkpoint_path,
        "--tasks",
        "*olmo3:dev:1b:vllm",
        "--priority",
        "urgent",
        "--cluster",
        "ai2/jupiter-cirrascale-2",
        "--num-gpus",
        "1",
        "--model-backend",
        "vllm",
        "--model-args",
        "dtype=bfloat16",
        "--dashboard",
        "regmixer",
        "--workspace",
        "ai2/dolma2",
        "--partition-size",
        "8",
        "--env-name",
        "cookbook-swarm-venv",
        "--no-push-datalake",
    ]

    # HF evaluation command
    hf_cmd = [
        "uv",
        "run",
        "olmo-cookbook-eval",
        "evaluate",
        checkpoint_path,
        "--tasks",
        "*olmo3:dev:1b:hf",
        "--priority",
        "urgent",
        "--cluster",
        "ai2/jupiter-cirrascale-2",
        "--num-gpus",
        "1",
        "--model-backend",
        "hf",
        "--model-args",
        "dtype=bfloat16",
        "--workspace",
        "ai2/dolma2",
        "--env-name",
        "cookbook-swarm-venv",
        "--no-push-datalake",
    ]

    return vllm_cmd, hf_cmd


def run_task(base_path: str, index: str, cmd_type: str, dry_run: bool = False):
    """Process a single checkpoint."""
    try:
        if cmd_type == "move":
            cmd = mk_move_ckpt(base_path, index)
            if dry_run:
                print(f"[DRY RUN] Would run: {' '.join(cmd)}")
            else:
                subprocess.run(cmd, check=True)
        elif cmd_type == "convert":
            cmd = mk_convert_ckpt(base_path, index)
            if dry_run:
                print(f"[DRY RUN] Would run: {' '.join(cmd)}")
            else:
                subprocess.run(cmd, check=True)
        elif cmd_type == "eval":
            vllm_cmd, hf_cmd = mk_eval_ckpt(base_path, index)
            if dry_run:
                print(f"[DRY RUN] Would run: {' '.join(vllm_cmd)}")
                print(f"[DRY RUN] Would run: {' '.join(hf_cmd)}")
            else:
                # Run both commands and wait for completion
                vllm_process = subprocess.Popen(vllm_cmd)
                hf_process = subprocess.Popen(hf_cmd)

                # Wait for both processes to complete
                vllm_process.wait()
                hf_process.wait()

                # Check return codes
                if vllm_process.returncode != 0:
                    raise subprocess.CalledProcessError(vllm_process.returncode, vllm_cmd)
                if hf_process.returncode != 0:
                    raise subprocess.CalledProcessError(hf_process.returncode, hf_cmd)
        else:
            raise ValueError(f"Unknown command type: {cmd_type}")

        print(f"Launched task {index} ({cmd_type})")

    except subprocess.CalledProcessError as e:
        print(f"Error processing checkpoint {index}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Batch process checkpoints")
    parser.add_argument(
        "--base-path",
        required=True,
        help="Base path for your swarm checkpoints (e.g., 'beaker_user/run_name')",
    )
    parser.add_argument(
        "--range",
        nargs="+",
        type=int,
        required=True,
        help="Range of indices to process. Single int for 0 to N-1, or two ints for start to end-1",
    )
    parser.add_argument(
        "--cmd-type", choices=["move", "convert", "eval"], required=True, help="Command type to run"
    )
    parser.add_argument("--max-workers", type=int, default=12, help="Maximum number of worker threads")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without executing them",
    )
    parser.add_argument(
        "--cancel-on-failure",
        action="store_true",
        help="Cancel remaining tasks if any task fails",
    )

    args = parser.parse_args()

    # Generate indices with 4-digit padding
    if len(args.range) == 1:
        # Single int: generate range from 0 to N-1
        indices = [f"{i:04d}" for i in range(args.range[0])]
    elif len(args.range) == 2:
        # Two ints: generate range from start to end-1
        indices = [f"{i:04d}" for i in range(args.range[0], args.range[1])]
    else:
        raise ValueError("Range must be either a single integer or two integers (start, end)")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(run_task, args.base_path, index, args.cmd_type, args.dry_run) for index in indices
        ]

        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed with error: {e}")
                    if args.cancel_on_failure:
                        print("Canceling remaining tasks due to failure...")
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False)
                        print("Canceled remaining tasks.")
                        return
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Canceling remaining tasks...")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False)
            print("Canceled remaining tasks.")
            return

    print("All tasks completed!")


if __name__ == "__main__":
    main()
