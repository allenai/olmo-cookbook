#!/usr/bin/env python3
"""
Tokenization orchestrator for downloading, tokenizing, and uploading data.

This script handles the complete pipeline:
1. Download data from remote storage using s5cmd
2. Tokenize the data locally using dolma tokens
3. Upload tokenized data back to remote storage
4. Report destination paths with appropriate wildcards

Usage:
    python tokenization_orchestrator.py <command> <input_file> [options]

Commands:
    download    - Download data from remote storage
    tokenize    - Tokenize downloaded data
    upload      - Upload tokenized data to remote storage
    destination_paths - Output final destination paths
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class DataProcessor:
    def __init__(self, input_file: str, remote_prefix: str = "s3://", 
                 input_prefix: str = "ai2-llm/pretraining-data/sources/",
                 output_prefix: str = "ai2-llm/preprocessed/",
                 local_dir: str = "/mnt/raid0",
                 tokenizer: str = "allenai/dolma2-tokenizer"):
        self.input_file = input_file
        # Don't strip slashes from remote_prefix as it needs to keep s3:// format
        self.remote_prefix = remote_prefix
        self.input_prefix = input_prefix.strip('/')
        self.output_prefix = output_prefix.strip('/')
        self.local_dir = Path(local_dir)
        self.tokenizer = tokenizer
        self.tokenizer_suffix = tokenizer.replace('/', '_')
        
        # Read input paths
        self.paths = self._read_input_paths()
        
    def _read_input_paths(self) -> List[str]:
        """Read and clean input paths from file."""
        with open(self.input_file, 'r') as f:
            paths = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove remote prefix if present
                    if line.startswith(self.remote_prefix):
                        line = line[len(self.remote_prefix):].lstrip('/')
                    paths.append(line)
        return paths
    
    def _run_command(self, cmd: List[str], description: str) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with exit code {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}"
            return False, error_msg
    
    def _check_disk_space(self) -> bool:
        """Check if there's at least 10TB of free disk space."""
        statvfs = os.statvfs(self.local_dir)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_tb = free_bytes / (1024**4)  # Convert to TB
        
        print(f"Available disk space: {free_tb:.2f} TB")
        if free_tb < 10:
            print(f"ERROR: Insufficient disk space. Need at least 10TB, have {free_tb:.2f}TB")
            return False
        return True
    
    def _get_first_jsonl_line(self, path: str) -> Optional[dict]:
        """Get the first line from a JSONL file to inspect field structure."""
        local_path = self.local_dir / path
        
        # Look for .jsonl.gz files first
        for file_path in local_path.rglob("*.jsonl.gz"):
            try:
                import gzip
                with gzip.open(file_path, 'rt') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        return json.loads(first_line)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
                
        # Look for .jsonl files
        for file_path in local_path.rglob("*.jsonl"):
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        return json.loads(first_line)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
                
        return None
    
    def _detect_id_field_issues(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect ID field name and type issues."""
        first_line = self._get_first_jsonl_line(path)
        if not first_line:
            return None, None
            
        # Check if 'id' field exists and its type
        if 'id' in first_line:
            id_type = type(first_line['id']).__name__
            if id_type == 'int':
                return 'id', 'int'
            return 'id', 'str'
        
        # Look for fields with 'id' in the name
        id_fields = [k for k in first_line.keys() if 'id' in k.lower()]
        if id_fields:
            field_name = id_fields[0]  # Use first match
            id_type = type(first_line[field_name]).__name__
            return field_name, 'int' if id_type == 'int' else 'str'
            
        return None, None
    
    def _remove_empty_files(self) -> int:
        """Remove empty .jsonl.gz and .jsonl files and return count of removed files."""
        removed_count = 0
        
        for path in self.paths:
            local_path = self.local_dir / path
            if not local_path.exists():
                continue
            
            # Find all .jsonl.gz and .jsonl files
            jsonl_files = list(local_path.rglob("*.jsonl.gz")) + list(local_path.rglob("*.jsonl"))
            
            for file_path in jsonl_files:
                if self._is_empty_jsonl_file(file_path):
                    print(f"Removing empty file: {file_path}")
                    file_path.unlink()
                    removed_count += 1
        
        return removed_count
    
    def _is_empty_jsonl_file(self, file_path: Path) -> bool:
        """Check if a JSONL file is empty (contains no valid JSON objects)."""
        try:
            if file_path.suffix == '.gz':
                import gzip
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                json.loads(line)
                                return False  # Found at least one valid JSON object
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON lines
            else:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                json.loads(line)
                                return False  # Found at least one valid JSON object
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON lines
            
            # If we get here, no valid JSON objects were found
            return True
            
        except Exception as e:
            print(f"Warning: Could not check file {file_path}: {e}")
            return False  # Don't remove files we can't read
    
    def download(self) -> bool:
        """Download data from remote storage."""
        print("=== DOWNLOADING DATA ===")
        errors = []
        
        for path in self.paths:
            remote_path = f"{self.remote_prefix}{path}"
            local_path = self.local_dir / path
            
            # Check if this is a file or directory based on the path
            if path.endswith(('.gz', '.jsonl', '.json')):
                # Single file - download directly
                local_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = ["s5cmd", "cp", remote_path, str(local_path)]
            else:
                # Directory - add wildcard for sync
                if not remote_path.endswith('/'):
                    remote_path += '/'
                remote_path += '*'
                
                local_path.mkdir(parents=True, exist_ok=True)
                cmd = ["s5cmd", "sync", remote_path, str(local_path) + "/"]
            
            success, output = self._run_command(cmd, f"Downloading {remote_path}")
            
            if not success:
                errors.append(f"Failed to download {remote_path}: {output}")
            else:
                print(f"Successfully downloaded {remote_path}")
        
        # Check for and remove empty files
        total_removed = self._remove_empty_files()
        
        if errors:
            print("\n=== DOWNLOAD ERRORS ===")
            for error in errors:
                print(error)
            return False
        
        print("All downloads completed successfully!")
        if total_removed > 0:
            print(f"Removed {total_removed} empty files")
        return True
    
    def tokenize(self) -> bool:
        """Tokenize the downloaded data."""
        print("=== TOKENIZING DATA ===")
        
        # Check disk space first
        if not self._check_disk_space():
            return False
        
        # Download tokenizer
        tokenizer_dir = self.local_dir / "tokenizer"
        if not tokenizer_dir.exists():
            cmd = ["uv", "run", "huggingface-cli", "download", self.tokenizer, "--local-dir", str(tokenizer_dir)]
            success, output = self._run_command(cmd, "Downloading tokenizer")
            if not success:
                print(f"Failed to download tokenizer: {output}")
                return False
        
        errors = []
        
        for path in self.paths:
            local_input_path = self.local_dir / path
            
            # Detect ID field issues
            id_field_name, id_field_type = self._detect_id_field_issues(path)
            
            # Determine if path has subdirectories or is a single file
            if path.endswith(('.gz', '.jsonl', '.json')):
                # Single file - use exact path for input, strip extension for output
                doc_path = str(local_input_path)
                # Remove extensions for destination
                dest_base = path
                for ext in ['.jsonl.gz', '.json.gz', '.gz', '.jsonl', '.json']:
                    if dest_base.endswith(ext):
                        dest_base = dest_base[:-len(ext)]
                        break
                dest_path = self.local_dir / f"{dest_base}_{self.tokenizer_suffix}"
            else:
                # Directory - check if it has immediate .gz files or subdirectories
                if list(local_input_path.glob("*.gz")):
                    # Has immediate .gz files
                    doc_path = f"{local_input_path}/*"
                    dest_path = self.local_dir / f"{path}_{self.tokenizer_suffix}"
                else:
                    # Has subdirectories - process each separately
                    subdirs = [d for d in local_input_path.iterdir() if d.is_dir()]
                    if subdirs:
                        # Process each subdirectory
                        for subdir in subdirs:
                            subdir_rel = path + "/" + subdir.name
                            sub_doc_path = f"{subdir}/*"
                            sub_dest_path = self.local_dir / f"{subdir_rel}_{self.tokenizer_suffix}"
                            
                            success = self._run_tokenization_command(
                                sub_doc_path, sub_dest_path, tokenizer_dir, 
                                id_field_name, id_field_type
                            )
                            if not success:
                                errors.append(f"Failed to tokenize {subdir_rel}")
                        continue
                    else:
                        # Empty directory or no subdirs, treat as regular
                        doc_path = f"{local_input_path}/*"
                        dest_path = self.local_dir / f"{path}_{self.tokenizer_suffix}"
            
            success = self._run_tokenization_command(
                doc_path, dest_path, tokenizer_dir, id_field_name, id_field_type
            )
            if not success:
                errors.append(f"Failed to tokenize {path}")
        
        if errors:
            print("\n=== TOKENIZATION ERRORS ===")
            for error in errors:
                print(error)
            return False
        
        print("All tokenization completed successfully!")
        return True
    
    def _run_tokenization_command(self, doc_path: str, dest_path: Path, 
                                 tokenizer_dir: Path, id_field_name: Optional[str], 
                                 id_field_type: Optional[str]) -> bool:
        """Run a single tokenization command."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build tokenization command
        cmd = [
            "uv", "run", "dolma", "tokens",
            "--documents", doc_path,
            "--destination", str(dest_path),
            "--tokenizer.name_or_path", str(tokenizer_dir / "tokenizer.json"),
            "--tokenizer.eos_token_id", "100257",
            "--tokenizer.pad_token_id", "100277",
            "--no-tokenizer.segment_before_tokenization",
            "--tokenizer.encode_special_tokens",
            "--processes", str(os.cpu_count()),
            "--max_size", "4_000_000_000",
            "--sample_ring_prop",
            "--dtype", "uint32"
        ]
        
        # Add ID field customizations if needed
        if id_field_name and id_field_name != 'id':
            cmd.extend(["--fields.id_field_name", id_field_name])
        
        if id_field_type == 'int':
            cmd.extend(["--fields.id_field_type", "int"])
        
        success, output = self._run_command(cmd, f"Tokenizing {doc_path}")
        if success:
            print(f"Successfully tokenized {doc_path}")
        else:
            print(f"Failed to tokenize {doc_path}: {output}")
        
        return success
    
    def upload(self) -> bool:
        """Upload tokenized data to remote storage."""
        print("=== UPLOADING DATA ===")
        errors = []
        
        for path in self.paths:
            # Find tokenized directory
            if path.endswith(('.gz', '.jsonl', '.json')):
                # Single file case
                dest_base = path
                for ext in ['.jsonl.gz', '.json.gz', '.gz', '.jsonl', '.json']:
                    if dest_base.endswith(ext):
                        dest_base = dest_base[:-len(ext)]
                        break
                local_tokenized_path = self.local_dir / f"{dest_base}_{self.tokenizer_suffix}"
            else:
                local_tokenized_path = self.local_dir / f"{path}_{self.tokenizer_suffix}"
            
            if not local_tokenized_path.exists():
                errors.append(f"Tokenized data not found: {local_tokenized_path}")
                continue
            
            # Determine remote destination path
            if path.startswith(self.input_prefix):
                relative_path = path[len(self.input_prefix):].lstrip('/')
            else:
                relative_path = path
            
            remote_dest = f"{self.remote_prefix}{self.output_prefix}/{relative_path}_{self.tokenizer_suffix}/"
            
            cmd = ["s5cmd", "sync", f"{local_tokenized_path}/", remote_dest]
            success, output = self._run_command(cmd, f"Uploading {local_tokenized_path}")
            
            if not success:
                errors.append(f"Failed to upload {local_tokenized_path}: {output}")
            else:
                print(f"Successfully uploaded to {remote_dest}")
        
        if errors:
            print("\n=== UPLOAD ERRORS ===")
            for error in errors:
                print(error)
            return False
        
        print("All uploads completed successfully!")
        return True
    
    def destination_paths(self) -> None:
        """Output final destination paths with appropriate wildcards."""
        print("=== DESTINATION PATHS ===")
        
        for path in self.paths:
            # Determine relative path from input prefix
            if path.startswith(self.input_prefix):
                relative_path = path[len(self.input_prefix):].lstrip('/')
            else:
                relative_path = path
            
            base_dest = f"{self.remote_prefix}{self.output_prefix}/{relative_path}_{self.tokenizer_suffix}"
            
            # Check if there are subdirectories in the original path
            local_path = self.local_dir / path
            has_subdirs = False
            
            if local_path.exists():
                has_subdirs = any(d.is_dir() for d in local_path.iterdir() if d.is_dir())
            
            # Add appropriate wildcard
            if has_subdirs:
                wildcard_path = f"{base_dest}/**/*.gz"
            else:
                wildcard_path = f"{base_dest}/*.gz"
            
            print(wildcard_path)


def main():
    parser = argparse.ArgumentParser(description="All-in-one data processing script")
    parser.add_argument("command", choices=["download", "tokenize", "upload", "destination_paths"],
                       help="Command to execute")
    parser.add_argument("input_file", help="File containing line-separated remote paths")
    parser.add_argument("--remote-prefix", default="s3://", 
                       help="Remote storage prefix (default: s3://)")
    parser.add_argument("--input-prefix", default="ai2-llm/pretraining-data/sources/",
                       help="Input prefix to strip from paths")
    parser.add_argument("--output-prefix", default="ai2-llm/preprocessed/",
                       help="Output prefix for processed data")
    parser.add_argument("--local-dir", default="/mnt/raid0",
                       help="Local directory for processing (default: /mnt/raid0)")
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer",
                       help="Tokenizer to use (default: allenai/dolma2-tokenizer)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Create processor instance
    processor = DataProcessor(
        input_file=args.input_file,
        remote_prefix=args.remote_prefix,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        local_dir=args.local_dir,
        tokenizer=args.tokenizer
    )
    
    # Execute command
    success = True
    if args.command == "download":
        success = processor.download()
    elif args.command == "tokenize":
        success = processor.tokenize()
    elif args.command == "upload":
        success = processor.upload()
    elif args.command == "destination_paths":
        processor.destination_paths()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
