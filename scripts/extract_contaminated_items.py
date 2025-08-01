#!/usr/bin/env python3
"""
Script to extract contaminated item IDs and task names from decon reports.

Traverses a directory of decon reports and extracts tuples of (doc_id, task_names)
for benchmark test items marked as contaminated.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def find_jsonl_files(root_dir: Path) -> List[Path]:
    """Find all .jsonl files in the directory tree."""
    jsonl_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(Path(root) / file)
    return jsonl_files


def extract_contaminated_items(jsonl_file: Path) -> List[Tuple[int, str]]:
    """
    Extract (doc_id, task_name) tuples from a single JSONL file.
    
    Returns:
        List of (doc_id, task_name) tuples where oe-eval-task is not null
    """
    items = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract the required fields
                    doc_id = data.get('oe-eval-doc-id')
                    task_data = data.get('oe-eval-task')
                    
                    # Skip if oe-eval-task is null or doc_id is missing
                    if task_data is not None and doc_id is not None:
                        # Handle both string and list formats for oe-eval-task
                        if isinstance(task_data, list):
                            # If it's a list, add each task separately
                            for task_name in task_data:
                                if task_name:  # Skip empty strings
                                    items.append((doc_id, task_name))
                        elif isinstance(task_data, str) and task_data:
                            # If it's a string, add it directly
                            items.append((doc_id, task_data))
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON in {jsonl_file}:{line_num}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error reading file {jsonl_file}: {e}")
        
    return items


def group_by_doc_id(items: List[Tuple[int, str]]) -> Dict[int, List[str]]:
    """
    Group task names by document ID.
    
    Args:
        items: List of (doc_id, task_name) tuples
        
    Returns:
        Dictionary mapping doc_id to list of unique task names
    """
    grouped = defaultdict(set)
    
    for doc_id, task_name in items:
        grouped[doc_id].add(task_name)
    
    # Convert sets to sorted lists
    return {doc_id: sorted(list(task_names)) for doc_id, task_names in grouped.items()}


def format_output(grouped_items: Dict[int, List[str]]) -> List[str]:
    """
    Format the output as tuples in the specified format.
    
    Args:
        grouped_items: Dictionary mapping doc_id to list of task names
        
    Returns:
        List of formatted strings representing tuples
    """
    output_lines = []
    
    # Sort by doc_id for consistent output
    for doc_id in sorted(grouped_items.keys()):
        task_names = grouped_items[doc_id]
        
        # Format task names as a list
        if len(task_names) == 1:
            task_list = f'["{task_names[0]}"]'
        else:
            task_list = '[' + ', '.join(f'"{task}"' for task in task_names) + ']'
        
        output_lines.append(f'({doc_id}, {task_list})')
    
    return output_lines


def main():
    parser = argparse.ArgumentParser(
        description="Extract contaminated item IDs and task names from decon reports"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory containing decon reports"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output file to write the list of tuples"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about processed files"
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    # Find all JSONL files
    print(f"Scanning directory: {args.input_dir}")
    jsonl_files = find_jsonl_files(args.input_dir)
    
    if args.stats:
        print(f"Found {len(jsonl_files)} JSONL files")
    
    # Extract contaminated items from all files
    all_items = []
    files_processed = 0
    
    for jsonl_file in jsonl_files:
        if args.stats:
            print(f"Processing: {jsonl_file}")
        
        items = extract_contaminated_items(jsonl_file)
        all_items.extend(items)
        files_processed += 1
        
        if args.stats and len(items) > 0:
            print(f"  Found {len(items)} contaminated items")
    
    # Group by document ID
    grouped_items = group_by_doc_id(all_items)
    
    # Format output
    output_lines = format_output(grouped_items)
    
    # Print statistics
    if args.stats:
        print(f"\nStatistics:")
        print(f"  Files processed: {files_processed}")
        print(f"  Total contaminated entries: {len(all_items)}")
        print(f"  Unique document IDs: {len(grouped_items)}")
        print(f"  Output lines: {len(output_lines)}")
    
    # Write output to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    print(f"Output written to: {args.output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
