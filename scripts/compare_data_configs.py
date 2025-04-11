#!/usr/bin/env python3
"""
This script compares the contents of multiple text files containing paths to .npy files.
It counts the number of .npy files under each base path and displays the results in a table.

Usage:
    # Compare mid-training annealing configs
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino*.txt
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino50.txt /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino100.txt

    # Compare pre-training data mixes
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/olmo_core/data/mixes/OLMoE-mix-0824.txt /Users/kylel/ai2/OLMo-core/src/olmo_core/data/mixes/dolma17.txt

    # Even compare old OLMo configs (YAML) with new ones (TXT)
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino50.txt /Users/kylel/ai2/OLMo/configs/official-1124/OLMo2-7B-stage2-seed*.yaml
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino100.txt /Users/kylel/ai2/OLMo/configs/official-1124/OLMo2-13B-stage2-seed*-100B.yaml
    python compare_data_configs.py /Users/kylel/ai2/OLMo-core/src/scripts/train/anneal/dolmino300.txt /Users/kylel/ai2/OLMo/configs/official-1124/OLMo2-13B-stage2-seed*-300B.yaml

    # Compare Wandb runs
    python compare_data_configs.py https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39 https://wandb.ai/ai2-llm/olmoe/runs/rzsn9tlc

Note that the counts are not file size; they are just number of `npy` files under each base path, which could be different sizes.
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


def read_config_files(path_patterns: List[str]) -> Dict[str, List[str]]:
    """
    Read paths from config files matching the provided patterns.
    Handles local files (.txt, .yaml) and Wandb URLs.

    Args:
        path_patterns: List of glob patterns or Wandb URLs

    Returns:
        Dictionary mapping filenames to lists of paths
    """
    paths_by_file: Dict[str, List[str]] = {}
    
    for pattern in path_patterns:
        if pattern.startswith(('http://', 'https://')):
            # Handle Wandb URLs
            import wandb
            api = wandb.Api()
            run_path = parse_wandb_run_path(pattern)
            try:
                run = api.run(run_path)
                config_raw = run._attrs["rawconfig"]
                
                # Extract paths from Wandb config
                data_paths = config_raw.get('data', {}).get('paths', [])
                paths = [path for path in data_paths if path.endswith('.npy')]
                
                # Use run ID as filename
                filename = f"wandb_{run.id}"
                paths_by_file[filename] = paths
            except Exception as e:
                print(f"Error reading Wandb config from {pattern}: {e}")
                continue
        else:
            # Handle local files as before
            files = glob.glob(pattern)
            for file in files:
                filename = Path(file).name
                paths = []

                if filename.endswith('.yaml'):
                    with open(file, 'r') as f:
                        config = yaml.safe_load(f)
                        data_paths = config.get('data', {}).get('paths', [])
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
                else:
                    with open(file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue

                            if ',' in line:
                                path = line.split(',')[1]
                            else:
                                path = line

                            if path.endswith('.npy'):
                                paths.append(path)

                paths_by_file[filename] = paths

    if not paths_by_file:
        print(f"No valid configs found in patterns: {path_patterns}")
        return {}

    return paths_by_file


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
    path = path.replace('\\', '/')
    
    # If the path appears to be a URL (e.g., starts with http, https, s3, or gs),
    # parse it and take only the path portion.
    if re.match(r'^(https?|s3|gs):', path, re.IGNORECASE):
        parsed = urlparse(path)
        path = parsed.path  # This drops the scheme and netloc
    
    # Remove any leading slashes for consistency
    return path.lstrip('/')

def count_paths(paths_by_file: Dict[str, List[str]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Count files under each base path for each config file after normalizing paths to treat variations
    (e.g. different prefixes) as the same base path.

    Args:
        paths_by_file: Dictionary mapping filenames to lists of paths

    Returns:
        Tuple of (base path counts dict, file totals dict)
    """
    base_path_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for filename, paths in paths_by_file.items():
        for path in paths:
            # Normalize the incoming path to remove extraneous prefixes
            norm_path = normalize_storage_path(path)
            
            # Special handling for dclm paths: extract the part up to the "tokenizer" folder.
            if '/dclm/' in norm_path.lower():
                match = re.match(r'(.*dclm/.+?tokenizer)(?:/.*)?$', norm_path, re.IGNORECASE)
                base_path = match.group(1) if match else os.path.normpath(norm_path)
            else:
                # For all other paths, group by the normalized full path (removing the last component).
                parts = norm_path.split('/')
                if len(parts) > 1:
                    base_path = os.path.normpath(os.path.join(*parts[:-1]))
                else:
                    base_path = norm_path  # If there's no directory structure, use the path as-is.
            
            base_path_counts[base_path][filename] += 1

    # Optionally, compute the total number of paths for each file.
    file_totals = {filename: len(paths) for filename, paths in paths_by_file.items()}
    return base_path_counts, file_totals

def format_results(base_path_counts: Dict[str, Dict[str, int]], 
                              file_totals: Dict[str, int]) -> None:
    """
    Format and print results using tabulate.
    
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
        # Primary sort by first column, secondary sort by second column
        return (count1, count2)
        
    sorted_base_paths = sorted(
        base_path_counts.keys(),
        key=sort_key,
        reverse=True
    )
    
    # Prepare headers and table data.
    # Truncate filenames that are too long (over 30 chars)
    truncated_filenames = []
    for filename in sorted_filenames:
        if len(filename) > 30:
            truncated = filename[:27] + "..."
        else:
            truncated = filename
        truncated_filenames.append(truncated)
    
    headers = ['Base Path'] + truncated_filenames + ['Match']
    table_data = []
    
    for base_path in sorted_base_paths:
        # Build a row for each base_path.
        row = [base_path]
        counts = [base_path_counts[base_path].get(filename, 0) for filename in sorted_filenames]
        # Use the count if greater than 0, else '-'
        row.extend([count if count > 0 else '-' for count in counts])
        
        # Add a checkmark if all counts are the same.
        all_same = True
        if counts: 
            first_val = counts[0]
            for count in counts:
                if count != first_val:
                    all_same = False
                    break
        else:
            all_same = False
        row.append('✓' if all_same else '')
        
        table_data.append(row)
    
    # Build totals row.
    totals_row = ['Total'] + [file_totals[filename] for filename in sorted_filenames] + ['']
    table_data.append(totals_row)
    
    # Print formatted table using tabulate.
    print("\n" + tabulate(table_data, headers, tablefmt='grid'))

def compare_config_files(path_patterns: List[str]) -> None:
    """
    Compare text files containing paths and print counts of .npy files under each base path.

    Args:
        path_patterns: List of glob patterns to match files
    """
    paths_by_file = read_config_files(path_patterns)
    if not paths_by_file:
        return

    base_path_counts, file_totals = count_paths(paths_by_file)
    format_results(base_path_counts, file_totals)


@click.command()
@click.argument('path_patterns', nargs=-1, required=True)
def main(path_patterns: Tuple[str, ...]) -> None:
    """Compare text files containing paths and print counts of .npy files under each base path."""
    compare_config_files(list(path_patterns))


if __name__ == "__main__":
    main()
