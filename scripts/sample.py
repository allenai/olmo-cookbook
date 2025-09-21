#!/usr/bin/env python3

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import random

# Global variable to store detailed sampling statistics
_detailed_sampling_stats = None


def get_token_count_from_file(npy_file: Path) -> int:
    """Get token count from a .npy file based on file size (assumes uint32 dtype)"""
    try:
        file_size = npy_file.stat().st_size
        # uint32 = 4 bytes per token
        return file_size // 4
    except Exception as e:
        print(f"Error processing {npy_file}: {e}")
        return 0


def collect_bucket_files(
    input_dir: Path, target_buckets: Optional[List[str]] = None
) -> Dict[str, List[Tuple[Path, int]]]:
    """Collect all .npy files from buckets with their token counts"""
    bucket_files = {}

    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue

        for bucket_dir in subdir.iterdir():
            if not bucket_dir.is_dir():
                continue

            bucket_name = bucket_dir.name

            # If target_buckets is specified, only process those buckets
            if target_buckets and bucket_name not in target_buckets:
                continue

            bucket_path = f"{subdir.name}/{bucket_name}"
            if bucket_path not in bucket_files:
                bucket_files[bucket_path] = []

            for npy_file in bucket_dir.glob("*.npy"):
                token_count = get_token_count_from_file(npy_file)
                bucket_files[bucket_path].append((npy_file, token_count))

    return bucket_files


def calculate_natural_distribution(
    bucket_files: Dict[str, List[Tuple[Path, int]]],
) -> Dict[str, float]:
    """Calculate the natural distribution of tokens across buckets"""
    total_tokens = 0
    bucket_tokens = {}

    for bucket_path, files in bucket_files.items():
        bucket_total = sum(token_count for _, token_count in files)
        bucket_tokens[bucket_path] = bucket_total
        total_tokens += bucket_total

    # Calculate proportions
    distribution = {}
    for bucket_path, tokens in bucket_tokens.items():
        distribution[bucket_path] = tokens / total_tokens if total_tokens > 0 else 0

    return distribution


def sample_files_by_budget(
    bucket_files: Dict[str, List[Tuple[Path, int]]],
    target_budget: int,
    bucket_ratios: Optional[List[float]] = None,
    target_buckets: Optional[List[str]] = None,
) -> List[Tuple[Path, Path]]:
    """Sample files based on token budget and optional bucket ratios"""

    # When target buckets are specified, sample from each category proportionally
    if target_buckets:
        if bucket_ratios and len(bucket_ratios) != len(target_buckets):
            raise ValueError(
                f"Number of ratios ({len(bucket_ratios)}) must match number of target buckets ({len(target_buckets)})"
            )

        # If no ratios provided, use equal distribution
        if not bucket_ratios:
            bucket_ratios = [1.0 / len(target_buckets)] * len(target_buckets)

        # Normalize ratios to sum to 1
        total_ratio = sum(bucket_ratios)
        bucket_ratios = [r / total_ratio for r in bucket_ratios]

        # Group by category first
        category_buckets = {}
        for bucket_path, files in bucket_files.items():
            category_name = bucket_path.split("/")[0]  # Extract category name
            bucket_name = bucket_path.split("/")[-1]  # Extract bucket name

            if category_name not in category_buckets:
                category_buckets[category_name] = {}
            category_buckets[category_name][bucket_name] = files

        # Calculate total tokens available in each category for target buckets
        category_totals = {}
        for category_name, buckets in category_buckets.items():
            total_tokens = 0
            for bucket_name in target_buckets:
                if bucket_name in buckets:
                    total_tokens += sum(
                        token_count for _, token_count in buckets[bucket_name]
                    )
            category_totals[category_name] = total_tokens

        # Calculate proportional budget for each category
        global_total = sum(category_totals.values())

        selected_files = []
        bucket_token_counts = {bucket_name: 0 for bucket_name in target_buckets}
        detailed_sampling_stats = {}

        for category_name, buckets in category_buckets.items():
            if category_totals[category_name] == 0:
                continue

            # Calculate this category's share of the total budget
            category_budget = int(
                target_budget * (category_totals[category_name] / global_total)
            )

            print(f"Category {category_name}: budget={category_budget:,} tokens")

            detailed_sampling_stats[category_name] = {
                "category_budget": category_budget,
                "buckets": {},
            }

            # Sample from each bucket in this category according to target ratios
            for i, bucket_name in enumerate(target_buckets):
                if bucket_name not in buckets:
                    continue

                bucket_category_budget = int(category_budget * bucket_ratios[i])

                if bucket_category_budget <= 0:
                    continue

                files = buckets[bucket_name]
                if not files:
                    continue

                # Sort files by token count (smallest first for better fitting)
                files.sort(key=lambda x: x[1])

                current_tokens = 0
                bucket_selected = []

                for file_path, token_count in files:
                    if current_tokens < bucket_category_budget:
                        # Always add the file if we're under budget, even if it pushes us over
                        rel_path = file_path.relative_to(
                            file_path.parents[2]
                        )  # Remove input_dir prefix
                        bucket_selected.append((file_path, rel_path))
                        current_tokens += token_count
                    else:
                        # We've met or exceeded the budget, stop adding files
                        break

                selected_files.extend(bucket_selected)
                bucket_token_counts[bucket_name] += current_tokens

                # Store detailed stats for this category/bucket combination
                detailed_sampling_stats[category_name]["buckets"][bucket_name] = {
                    "target_budget": bucket_category_budget,
                    "actual_tokens": current_tokens,
                    "files_selected": len(bucket_selected),
                    "ratio": bucket_ratios[i],
                }

                print(
                    f"  {category_name}/{bucket_name}: selected {len(bucket_selected)} files, {current_tokens:,} tokens"
                )

        print("\nFinal bucket totals:")
        for bucket_name, total_tokens in bucket_token_counts.items():
            print(f"Bucket {bucket_name}: {total_tokens:,} tokens")

        # Store detailed stats in a global variable for access in main()
        global _detailed_sampling_stats
        _detailed_sampling_stats = detailed_sampling_stats
        return selected_files

    else:
        # Original logic for when no target buckets specified
        if bucket_ratios and len(bucket_ratios) != len(bucket_files):
            raise ValueError(
                f"Number of ratios ({len(bucket_ratios)}) must match number of buckets ({len(bucket_files)})"
            )

        # If no ratios provided, use equal distribution
        if not bucket_ratios:
            bucket_ratios = [1.0 / len(bucket_files)] * len(bucket_files)

        # Normalize ratios to sum to 1
        total_ratio = sum(bucket_ratios)
        bucket_ratios = [r / total_ratio for r in bucket_ratios]

        selected_files = []

        for i, (bucket_path, files) in enumerate(bucket_files.items()):
            bucket_budget = int(target_budget * bucket_ratios[i])

            if bucket_budget <= 0:
                continue

            # Sort files by token count (smallest first for better fitting)
            files.sort(key=lambda x: x[1])

            current_tokens = 0
            bucket_selected = []

            for file_path, token_count in files:
                if current_tokens < bucket_budget:
                    # Always add the file if we're under budget, even if it pushes us over
                    rel_path = file_path.relative_to(
                        file_path.parents[2]
                    )  # Remove input_dir prefix
                    bucket_selected.append((file_path, rel_path))
                    current_tokens += token_count
                else:
                    # We've met or exceeded the budget, stop adding files
                    break

            selected_files.extend(bucket_selected)
            print(
                f"Bucket {bucket_path}: selected {len(bucket_selected)} files, {current_tokens:,} tokens"
            )

        return selected_files


def copy_sampled_files(selected_files: List[Tuple[Path, Path]], output_dir: Path):
    """Copy selected files to output directory maintaining structure"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for src_path, rel_path in selected_files:
        dst_path = output_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Copy the .npy file
            shutil.copy2(src_path, dst_path)

            # Copy corresponding .csv.gz manifest file if it exists
            manifest_src = src_path.with_suffix(".csv.gz")
            if manifest_src.exists():
                manifest_dst = dst_path.with_suffix(".csv.gz")
                shutil.copy2(manifest_src, manifest_dst)

        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample tokens from buckets based on budget and ratios",
        epilog="Example: python sample_token_buckets.py --input-dir input_dir --output-dir output_dir --budget 1000000 --buckets bucket_0019 bucket_0017 --ratios 0.3 0.7",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Input directory containing subdirs with buckets",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, help="Output directory for sampled files"
    )
    parser.add_argument(
        "--budget", "-b", type=int, required=True, help="Target token budget"
    )
    parser.add_argument(
        "--buckets", "-k", nargs="+", help="Specific bucket names to sample from"
    )
    parser.add_argument(
        "--ratios",
        "-r",
        nargs="+",
        type=float,
        help="Sampling ratios for buckets (must match bucket count)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target budget: {args.budget:,} tokens")
    if args.buckets:
        print(f"Target buckets: {args.buckets}")
    if args.ratios:
        print(f"Bucket ratios: {args.ratios}")

    # Collect all bucket files
    print("Collecting bucket files...")
    bucket_files = collect_bucket_files(
        input_dir, args.buckets if args.buckets else None
    )

    if not bucket_files:
        print("No bucket files found")
        return 1

    print(f"Found {len(bucket_files)} buckets:")
    total_tokens = 0
    for bucket_path, files in bucket_files.items():
        bucket_tokens = sum(token_count for _, token_count in files)
        total_tokens += bucket_tokens
        print(f"  {bucket_path}: {len(files)} files, {bucket_tokens:,} tokens")

    print(f"Total available tokens: {total_tokens:,}")

    if args.budget > total_tokens:
        print(
            f"Warning: Budget ({args.budget:,}) exceeds available tokens ({total_tokens:,})"
        )

    # Calculate natural distribution if no ratios provided
    if not args.ratios:
        natural_dist = calculate_natural_distribution(bucket_files)
        print("Natural distribution:")
        for bucket_path, ratio in natural_dist.items():
            print(f"  {bucket_path}: {ratio:.3f}")

    # Sample files based on budget and ratios
    print("Sampling files...")
    selected_files = sample_files_by_budget(
        bucket_files,
        args.budget,
        args.ratios if args.ratios else None,
        args.buckets if args.buckets else None,
    )

    if not selected_files:
        print("No files selected")
        return 1

    total_sampled_tokens = sum(
        get_token_count_from_file(src) for src, _ in selected_files
    )
    print(f"Selected {len(selected_files)} files with {total_sampled_tokens:,} tokens")

    # Copy files to output directory
    print("Copying files...")
    copy_sampled_files(selected_files, output_dir)

    print(f"Sampling complete! Files copied to {output_dir}")

    # Write summary
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_budget": args.budget,
        "actual_tokens": total_sampled_tokens,
        "files_selected": len(selected_files),
        "buckets_sampled": list(bucket_files.keys()),
        "ratios_used": args.ratios,
        "target_buckets": args.buckets,
    }

    # Add detailed sampling statistics if available
    if _detailed_sampling_stats is not None:
        summary["detailed_sampling"] = _detailed_sampling_stats

        # Also add bucket totals summary
        bucket_totals = {}
        for category_name, category_data in _detailed_sampling_stats.items():
            for bucket_name, bucket_data in category_data["buckets"].items():
                if bucket_name not in bucket_totals:
                    bucket_totals[bucket_name] = {
                        "total_tokens": 0,
                        "total_files": 0,
                        "target_ratio": bucket_data["ratio"],
                    }
                bucket_totals[bucket_name]["total_tokens"] += bucket_data[
                    "actual_tokens"
                ]
                bucket_totals[bucket_name]["total_files"] += bucket_data[
                    "files_selected"
                ]

        summary["bucket_totals"] = bucket_totals

    summary_path = output_dir / "sampling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    exit(main())
