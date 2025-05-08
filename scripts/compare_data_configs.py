#!/usr/bin/env python3
"""
This script compares the contents of multiple text files containing paths to .npy files.
It counts the number of .npy files under each base path and displays the results in a table.

Usage:
    # Compare mid-training annealing configs
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino*.txt --data-only
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino50.txt /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino100.txt --data-only

    # Compare pre-training data mixes
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/olmo_core/data/mixes/OLMoE-mix-0824.txt /Users/kylel/ai2/OLMo-core/src/olmo_core/data/mixes/dolma17.txt --data-only

    # Even compare old OLMo configs (YAML) with new ones (TXT)
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino50.txt /Users/kylel/ai2/OLMo/configs/official-1124/OLMo2-7B-stage2-seed*.yaml --data-only
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino100.txt /Users/kylel/ai2/OLMo/configs/official-1124/OLMo2-13B-stage2-seed*-100B.yaml --data-only
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino300.txt /Users/kylel/ai2/OLMo/configs/official-1124/OLMo2-13B-stage2-seed*-300B.yaml --data-only

    # Compare Wandb runs
    python compare_data_configs.py https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39 https://wandb.ai/ai2-llm/olmoe/runs/rzsn9tlc

    # Compare a Wandb run on old OLMo configs with local files with new OLMo configs
    python compare_data_configs.py https://wandb.ai/ai2-llm/olmo-medium/runs/yh8qbwif\?nw\=nwusersoldni /Users/kylel/ai2/olmo-cookbook/peteish7-anneal-from-928646-50B-no-opt-repro__data_paths.txt  /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino50.txt --data-only
    python compare_data_configs.py https://wandb.ai/ai2-llm/olmo-medium/runs/79h5h3aa\?nw\=nwusersoldni /Users/kylel/ai2/olmo-cookbook/peteish7-anneal-from-928646-50B-no-opt-repro__data_paths.txt  /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino50.txt --data-only

    # Let's say we want to do a large scale comparison of all the configs of a single type of model (e.g. OLMo 1b)
    python scripts/compare_data_configs.py \
        https://wandb.ai/ai2-llm/olmo-cookbook/runs/fa12nu26 \
        https://wandb.ai/ai2-llm/olmo-cookbook/runs/qmfgv9bg \
        https://wandb.ai/ai2-llm/olmo-cookbook/runs/ezyubhfe/overview \
        https://wandb.ai/ai2-llm/olmo-cookbook/runs/w36osax9/overview \
        https://wandb.ai/ai2-llm/olmo-cookbook/runs/5v2r3oma \
        https://wandb.ai/ai2-llm/olmo-cookbook/runs/dqp8dbpp \
        --full-config


Note that the counts are not file size; they are just number of `npy` files under each base path, which could be different sizes.

@kylel

"""

import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import click
import yaml
from tabulate import tabulate


def parse_wandb_run_path(run_path: str) -> str:
    """
    Parse a Weights & Biases run path, either in the form "entity/project/run_id"
    or as a full wandb.ai URL.

    Args:
        run_path: Either a W&B run path like "ai2-llm/olmo-medium/cej4ya39"
                 or URL like "https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39"
                 or URL with username like "https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39?nw=nwusersoldni"

    Returns:
        Standardized run path in the form "entity/project/run_id"
        If input is not a W&B path/URL, returns the original string unchanged
    """
    run_path = run_path.strip("/")
    run_path_re = re.compile(r"^[^/]+/[^/]+/[^/]+$")
    run_path_url = re.compile(r"^https?://wandb.ai/([^/]+)/([^/]+)/runs/([^/?]+)")

    if run_path_re.match(run_path):
        return run_path

    m = run_path_url.match(run_path)
    if m is not None:
        entity, project, run_id = m.groups()
        return f"{entity}/{project}/{run_id}"

    return run_path  # Return original path if it's not a W&B path


def read_config_files(path_patterns: List[str], data_only: bool = True) -> Dict[str, Dict]:
    """
    Read paths from config files matching the provided patterns.
    Handles local files (.txt, .yaml) and Wandb URLs.

    Args:
        path_patterns: List of glob patterns or Wandb URLs
        data_only: If True, only extract data paths. If False, extract all config keys.

    Returns:
        Dictionary mapping filenames to either lists of paths or full config dicts
    """
    configs_by_file: Dict[str, Dict] = {}
    
    for pattern in path_patterns:
        if pattern.startswith(("http://", "https://")):
            # Handle Wandb URLs
            import wandb

            api = wandb.Api()
            run_path = parse_wandb_run_path(pattern)
            try:
                run = api.run(run_path)
                config_raw = run._attrs["rawconfig"]
                
                filename = f"wandb_{run.id}"
                if data_only:
                    # Extract just the data paths
                    data_paths = config_raw.get('data', {}).get('paths', [])
                    paths = [path for path in data_paths if path.endswith('.npy')]
                    configs_by_file[filename] = {'paths': paths}
                else:
                    # Include full config
                    configs_by_file[filename] = config_raw
            except Exception as e:
                print(f"Error reading Wandb config from {pattern}: {e}")
                continue
        else:
            # Handle local files
            files = glob.glob(pattern)
            for file in files:
                filename = Path(file).name
                
                if filename.endswith('.yaml'):
                    with open(file, 'r') as f:
                        config = yaml.safe_load(f)
                        if data_only:
                            data_paths = config.get('data', {}).get('paths', [])
                            paths = []
                            for line in data_paths:
                                line = line.strip()
                                if not line or line.startswith('#'):
                                    continue
                                if ',' in line:
                                    path = line.split(',')[1]
                                else:
                                    path = line
                                if path.endswith('.npy'):
                                    paths.append(path)
                            configs_by_file[filename] = {'paths': paths}
                        else:
                            configs_by_file[filename] = config
                else:
                    paths = []
                    with open(file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if ',' in line:
                                path = line.split(',')[1]
                            else:
                                path = line
                            if path.endswith('.npy'):
                                paths.append(path)
                    configs_by_file[filename] = {'paths': paths}

    if not configs_by_file:
        print(f"No valid configs found in patterns: {path_patterns}")
        return {}

    return configs_by_file


def normalize_storage_path(path: str) -> str:
    """
    Normalize a file path by converting backslashes to forward slashes,
    stripping known URL schemes and domains, and removing leading slashes.

    Args:
        path: The original file path.

    Returns:
        The normalized file path.
    """
    # Replace backslashes with forward slashes
    path = path.replace("\\", "/")

    # If the path appears to be a URL (e.g., starts with http, https, s3, or gs),
    # parse it and take only the path portion.
    if re.match(r"^(https?|s3|gs):", path, re.IGNORECASE):
        parsed = urlparse(path)
        path = parsed.path  # This drops the scheme and netloc

    # Remove any leading slashes for consistency
    return path.lstrip("/")


def count_paths(configs_by_file: Dict[str, Dict]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Count files under each base path for each config file after normalizing paths to treat variations
    (e.g. different prefixes) as the same base path.

    Args:
        configs_by_file: Dictionary mapping filenames to config dicts containing paths

    Returns:
        Tuple of (base path counts dict, file totals dict)
    """
    base_path_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for filename, config in configs_by_file.items():
        paths = config.get('paths', [])
        for path in paths:
            # Normalize the incoming path to remove extraneous prefixes
            norm_path = normalize_storage_path(path)

            # Special handling for dclm paths: extract the part up to the "tokenizer" folder.
            if "/dclm/" in norm_path.lower():
                match = re.match(r"(.*dclm/.+?tokenizer)(?:/.*)?$", norm_path, re.IGNORECASE)
                base_path = match.group(1) if match else os.path.normpath(norm_path)
            else:
                # For all other paths, group by the normalized full path (removing the last component).
                parts = norm_path.split("/")
                if len(parts) > 1:
                    base_path = os.path.normpath(os.path.join(*parts[:-1]))
                else:
                    base_path = norm_path  # If there's no directory structure, use the path as-is.

            base_path_counts[base_path][filename] += 1

    # Compute the total number of paths for each file.
    file_totals = {filename: len(config.get('paths', [])) for filename, config in configs_by_file.items()}
    return base_path_counts, file_totals

def format_data_results(base_path_counts: Dict[str, Dict[str, int]], 
                       file_totals: Dict[str, int]) -> None:
    """
    Format and print results for data paths using tabulate.
    
    Args:
        base_path_counts: Dictionary mapping base paths to counts per file.
        file_totals: Dictionary mapping filenames to total counts.
    """
    # Sort filenames by their totals in descending order.
    sorted_filenames = sorted(file_totals.keys(), key=lambda x: file_totals[x], reverse=True)

    # Sort base paths by counts in the largest file, using second largest file as tiebreaker
    largest_file = sorted_filenames[0]
    second_largest = sorted_filenames[1] if len(sorted_filenames) > 1 else largest_file

    def sort_key(base_path):
        count1 = base_path_counts[base_path].get(largest_file, 0)
        count2 = base_path_counts[base_path].get(second_largest, 0)
        return (count1, count2)
        
    sorted_base_paths = sorted(base_path_counts.keys(), key=sort_key, reverse=True)
    
    # Prepare headers and table data.
    truncated_filenames = []
    for filename in sorted_filenames:
        truncated = filename[:27] + "..." if len(filename) > 30 else filename
        truncated_filenames.append(truncated)

    headers = ["Base Path"] + truncated_filenames + ["Match"]
    table_data = []

    for base_path in sorted_base_paths:
        row = [base_path]
        counts = [base_path_counts[base_path].get(filename, 0) for filename in sorted_filenames]
        row.extend([count if count > 0 else '-' for count in counts])
        
        all_same = len(set(c for c in counts if c > 0)) <= 1
        row.append('✓' if all_same else '')
        
        table_data.append(row)

    # Build totals row.
    totals_row = ["Total"] + [file_totals[filename] for filename in sorted_filenames] + [""]
    
    totals_row = ['Total'] + [file_totals[filename] for filename in sorted_filenames] + ['']
    table_data.append(totals_row)
    
    print("\n" + tabulate(table_data, headers, tablefmt='grid'))
    
def format_config_results(configs_by_file: Dict[str, Dict]) -> None:
    """
    Format and print results for non-data config keys, showing only keys with different values
    or keys that only exist in some configs.
    
    Args:
        configs_by_file: Dictionary mapping filenames to full config dicts
    """
    # Get all unique config keys across all files
    all_keys = set()
    for config in configs_by_file.values():
        all_keys.update(flatten_dict(config).keys())
    
    # Sort keys alphabetically
    sorted_keys = sorted(all_keys)
    filenames = list(configs_by_file.keys())
    
    # Print header
    print("\nConfig differences:")
    print("-" * 80)
    
    # Check each key for differences
    found_diffs = False
    for key in sorted_keys:
        # Skip data paths and dataset source mixture configs
        if ('data.paths' in key or 
            key == 'data' or 
            'dataset.source_mixture_config.source_configs' in key):
            continue
            
        values = []
        for filename in filenames:
            flat_config = flatten_dict(configs_by_file[filename])
            value = flat_config.get(key, '-')
            values.append(str(value))
        
        # Check if key exists in all configs
        exists_in_all = all(v != '-' for v in values)
        
        # Print if values are different or key doesn't exist in all configs
        unique_values = set(str(v) for v in values if v != '-')
        if len(unique_values) > 1 or not exists_in_all:
            found_diffs = True
            print(f"\n{key}:")
            for filename, value in zip(filenames, values):
                print(f"  {filename}: {value}")
    
    if not found_diffs:
        print("No differences found in config values.")

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten a nested dictionary into a single level dictionary with dot-separated keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compare_config_files(path_patterns: List[str], data_only: bool = True) -> None:
    """
    Compare config files and print either data path counts or full config comparisons.

    Args:
        path_patterns: List of glob patterns to match files
        data_only: If True, only compare data paths. If False, compare all config keys.
    """
    configs_by_file = read_config_files(path_patterns, data_only)
    if not configs_by_file:
        return

    if data_only:
        base_path_counts, file_totals = count_paths(configs_by_file)
        format_data_results(base_path_counts, file_totals)
    else:
        format_config_results(configs_by_file)


@click.command()
@click.argument('path_patterns', nargs=-1, required=True)
@click.option('--data-only/--full-config', default=True, 
              help='Compare only data paths (default) or all config keys')
def main(path_patterns: Tuple[str, ...], data_only: bool) -> None:
    """Compare config files and print either data path counts or full config comparisons.
    
    Example with flags:
        python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino*.txt --data-only
    """
    compare_config_files(list(path_patterns), data_only)


if __name__ == "__main__":
    main()
