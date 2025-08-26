#!/usr/bin/env python3
"""
Script to move tokenized data to the correct output locations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple


def read_input_paths(input_file: str) -> List[str]:
    """Read and clean input paths from file."""
    with open(input_file, 'r') as f:
        paths = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove s3:// prefix if present
                if line.startswith('s3://'):
                    line = line[5:].lstrip('/')
                paths.append(line)
    return paths


def find_tokenized_directories(local_dir: Path, tokenizer_suffix: str) -> List[Tuple[Path, str]]:
    """Find all directories ending with tokenizer suffix and their original paths."""
    found_dirs = []
    
    # Look for directories ending with the tokenizer suffix
    for root, dirs, files in os.walk(local_dir):
        for dir_name in dirs:
            if dir_name.endswith(tokenizer_suffix):
                full_path = Path(root) / dir_name
                # Extract the original path by removing the suffix
                original_path = dir_name[:-len(tokenizer_suffix)].rstrip('_')
                found_dirs.append((full_path, original_path))
    
    return found_dirs


def get_correct_destination(original_input_path: str, tokenized_dir_name: str, 
                          input_prefix: str, output_prefix: str, local_dir: Path) -> Path:
    """Calculate the correct destination path for tokenized data."""
    # Remove input prefix from the original path
    if original_input_path.startswith(input_prefix):
        relative_path = original_input_path[len(input_prefix):].lstrip('/')
    else:
        relative_path = original_input_path
    
    # Handle different cases based on the tokenized directory name
    if tokenized_dir_name == '':
        # This happens when the original was a directory and got "_allenai_dolma2-tokenizer" appended
        dest_path = local_dir / output_prefix / relative_path / "allenai" / "dolma2-tokenizer"
    elif '/' in tokenized_dir_name:
        # This handles subdirectory cases like "fim_50pct_psm_50pct/C"
        dest_path = local_dir / output_prefix / relative_path / tokenized_dir_name / "allenai" / "dolma2-tokenizer"
    else:
        # This handles file cases like "tulu-3-midtrain-v0-data-simple-concat-with-new-line-with-generation-prompt"
        dest_path = local_dir / output_prefix / relative_path / tokenized_dir_name / "allenai" / "dolma2-tokenizer"
    
    return dest_path


def move_tokenized_data(input_file: str, input_prefix: str, output_prefix: str, 
                       local_dir: str = "/mnt/raid0", tokenizer: str = "allenai/dolma2-tokenizer", dry_run: bool = False):
    """Move all tokenized data to correct locations."""
    local_dir_path = Path(local_dir)
    tokenizer_suffix = tokenizer.replace('/', '_')
    
    # Read original input paths
    input_paths = read_input_paths(input_file)
    
    # Find all tokenized directories
    tokenized_dirs = find_tokenized_directories(local_dir_path, tokenizer_suffix)
    
    print(f"Found {len(tokenized_dirs)} tokenized directories to move")
    
    moves_completed = []
    errors = []
    
    for tokenized_path, tokenized_name in tokenized_dirs:
        try:
            # Find matching input path
            matching_input = None
            for input_path in input_paths:
                # Check if this tokenized directory corresponds to this input path
                input_relative = input_path
                if input_path.startswith(input_prefix):
                    input_relative = input_path[len(input_prefix):].lstrip('/')
                
                # Check various patterns to match tokenized dir to input
                if (tokenized_path.parent.name in input_relative or 
                    input_relative in str(tokenized_path) or
                    any(part in str(tokenized_path) for part in input_relative.split('/')[-3:])):
                    matching_input = input_path
                    break
            
            if not matching_input:
                # Try to infer from the path structure
                path_parts = str(tokenized_path.relative_to(local_dir_path)).split('/')
                if len(path_parts) >= 3:
                    # Reconstruct likely input path
                    potential_input = '/'.join(path_parts[:-1])  # Remove the tokenized dir name
                    if potential_input.endswith('_' + tokenizer_suffix.split('_')[0]):
                        potential_input = potential_input[:-len('_' + tokenizer_suffix.split('_')[0])]
                    matching_input = input_prefix + potential_input
            
            if not matching_input:
                errors.append(f"Could not find matching input path for {tokenized_path}")
                continue
            
            # Calculate correct destination
            dest_path = get_correct_destination(matching_input, tokenized_name, 
                                              input_prefix, output_prefix, local_dir_path)
            
            if not dry_run:
                # Create destination directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Moving {tokenized_path} -> {dest_path}")
            
            # Move the directory
            if dest_path.exists():
                # print(f"Warning: Destination {dest_path} already exists, removing it first")
                # shutil.rmtree(dest_path)
                raise FileExistsError(f"Destination {dest_path} already exists.")
            if not dry_run:
                shutil.move(str(tokenized_path), str(dest_path))
            moves_completed.append((str(tokenized_path), str(dest_path)))
            
        except Exception as e:
            errors.append(f"Failed to move {tokenized_path}: {e}")
    
    # Report results
    print(f"\n=== MOVE RESULTS ===")
    print(f"Successfully moved {len(moves_completed)} directories:")
    for src, dst in moves_completed:
        print(f"  {src} -> {dst}")
    
    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for error in errors:
            print(f"  {error}")
        return False
    
    print("\nAll moves completed successfully!")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Move tokenized data to correct locations")
    parser.add_argument("input_file", help="Original input file with paths")
    parser.add_argument("--input-prefix", required=True, help="Input prefix used in tokenization")
    parser.add_argument("--output-prefix", required=True, help="Output prefix for correct locations")
    parser.add_argument("--local-dir", default="/mnt/raid0", help="Local directory")
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer", help="Tokenizer used")
    parser.add_argument("--dry-run", action='store_true', help="Perform a dry run without moving files")
    
    args = parser.parse_args()
    
    success = move_tokenized_data(
        args.input_file,
        args.input_prefix,
        args.output_prefix,
        args.local_dir,
        args.tokenizer,
        args.dry_run
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()