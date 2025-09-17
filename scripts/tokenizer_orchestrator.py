#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "dolma>=0.9.0",
#   "huggingface-hub[hf-transfer]>=0.34,<0.35",
#   "boto3",
#   "gcsfs",
#   "s3fs",
#   "requests",
#   "rich",
#   "platformdirs",
#   "pydantic",
#   "smart_open",
#   "yaspin",
#   "PyYAML>=6.0,<7.0",
#   "paramiko>=3.5,<3.6",
#   "tabulate",
#   "packaging>=24.2",
#   "tqdm>=4.67.1",
# ]
# ///
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
                 tokenizer: str = "allenai/dolma2-tokenizer",
                 dry_run: bool = False):
        self.input_file = input_file
        # Don't strip slashes from remote_prefix as it needs to keep s3:// format
        self.remote_prefix = remote_prefix
        self.input_prefix = input_prefix.strip('/')
        self.output_prefix = output_prefix.strip('/')
        self.local_dir = Path(local_dir)
        self.tokenizer = tokenizer
        self.tokenizer_suffix = tokenizer.replace('/', '_')
        self.dry_run = dry_run
        
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
        
        if self.dry_run:
            print("DRY RUN: Command would be executed here")
            return True, ""
        
        try:
            result = subprocess.run(cmd, text=True, check=True, env=os.environ.copy())
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"Command failed with exit code {e.returncode}"
    
    def _check_gcs_auth(self) -> bool:
        """Check if GCS authentication is properly configured."""
        try:
            result = subprocess.run(["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], 
                                  capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except:
            return False
    
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
        
        # Look for fields with 'id' in the name (case insensitive)
        id_fields = [k for k in first_line.keys() if 'id' in k.lower()]
        if id_fields:
            field_name = id_fields[0]  # Use first match
            id_type = type(first_line[field_name]).__name__
            print(f"Found ID field '{field_name}' of type {id_type} in {path}")
            return field_name, 'int' if id_type == 'int' else 'str'
        
        # Fallback: return empty string to disable ID field
        print(f"No ID field found in {path}, disabling ID field. Available fields: {list(first_line.keys())}")
        return "", None
    
    def _remove_empty_files(self) -> Tuple[int, List[str]]:
        """Remove empty .jsonl.gz and .jsonl files and return count and paths of removed files."""
        removed_count = 0
        removed_paths = []
        
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
                    removed_paths.append(str(file_path))
        
        return removed_count, removed_paths
    
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
        if self.dry_run:
            print("DRY RUN MODE: Showing what would be downloaded")
        
        errors = []
        
        for path in self.paths:
            remote_path = f"{self.remote_prefix}{path}"
            local_path = self.local_dir / path
            
            # Check if this is a file or directory based on the path
            if path.endswith(('.gz', '.jsonl', '.json')):
                # Single file - download directly
                if self.dry_run:
                    print(f"Would download file: {remote_path} -> {local_path}")
                else:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = ["s5cmd", "cp", "-sp", remote_path, str(local_path)]
            else:
                # Directory - add wildcard for sync
                if not remote_path.endswith('/'):
                    remote_path += '/'
                remote_path += '*'
                
                if self.dry_run:
                    print(f"Would sync directory: {remote_path} -> {local_path}/")
                else:
                    local_path.mkdir(parents=True, exist_ok=True)
                cmd = ["s5cmd", "sync", remote_path, str(local_path) + "/"]
            
            success, output = self._run_command(cmd, f"Downloading {remote_path}")
            
            if not success:
                errors.append(f"Failed to download {remote_path}: {output}")
            else:
                print(f"Successfully downloaded {remote_path}")
        
        if not self.dry_run:
            # Check for and remove empty files
            total_removed, removed_paths = self._remove_empty_files()
            
            if total_removed > 0:
                print(f"Removed {total_removed} empty files:")
                for path in removed_paths:
                    print(f"  {path}")
        
        if errors:
            print("\n=== DOWNLOAD ERRORS ===")
            for error in errors:
                print(error)
            return False
        
        print("All downloads completed successfully!")
        return True
    
    def tokenize(self) -> bool:
        """Tokenize the downloaded data."""
        print("=== TOKENIZING DATA ===")
        if self.dry_run:
            print("DRY RUN MODE: Showing what would be tokenized")
        
        # Check disk space first (skip in dry run)
        if not self.dry_run and not self._check_disk_space():
            return False
        
        # Download tokenizer (skip in dry run)
        tokenizer_dir = self.local_dir / "tokenizer"
        if not self.dry_run and not tokenizer_dir.exists():
            cmd = ["huggingface-cli", "download", self.tokenizer, "--local-dir", str(tokenizer_dir)]
            success, output = self._run_command(cmd, "Downloading tokenizer")
            if not success:
                print(f"Failed to download tokenizer: {output}")
                return False
        
        errors = []
        
        for path in self.paths:
            local_input_path = self.local_dir / path
            
            # Calculate relative path by removing input prefix
            if path.startswith(self.input_prefix):
                relative_path = path[len(self.input_prefix):].lstrip('/')
            else:
                relative_path = path
            
            # Detect ID field issues (skip in dry run)
            id_field_name, id_field_type = None, None
            if not self.dry_run:
                id_field_name, id_field_type = self._detect_id_field_issues(path)
            
            # Determine if path has subdirectories or is a single file
            if path.endswith(('.gz', '.jsonl', '.json')):
                # Single file - use exact path for input
                doc_path = str(local_input_path)
                # Remove extensions for destination path calculation
                dest_base = relative_path
                for ext in ['.jsonl.gz', '.json.gz', '.gz', '.jsonl', '.json']:
                    if dest_base.endswith(ext):
                        dest_base = dest_base[:-len(ext)]
                        break
                # Create proper output path structure
                dest_path = self.local_dir / self.output_prefix / dest_base / self.tokenizer.replace('/', '/')
                
                if self.dry_run:
                    print(f"Would tokenize file:")
                    print(f"  Input:  {doc_path}")
                    print(f"  Output: {dest_path}")
                
            else:
                # Directory - check if it has immediate .gz files or subdirectories
                has_immediate_gz = False
                subdirs = []
                
                # Always check filesystem for proper detection (even in dry run)
                if local_input_path.exists():
                    has_immediate_gz = bool(list(local_input_path.glob("*.gz")))
                    subdirs = [d for d in local_input_path.iterdir() if d.is_dir()]
                
                if has_immediate_gz:
                    # Has immediate .gz files - tokenize whole directory
                    doc_path = f"{local_input_path}/*"
                    dest_path = self.local_dir / self.output_prefix / relative_path / self.tokenizer.replace('/', '/')
                    
                    if self.dry_run:
                        print(f"Would tokenize directory with immediate .gz files:")
                        print(f"  Input:  {doc_path}")
                        print(f"  Output: {dest_path}")
                    
                elif subdirs:
                    # Has subdirectories - process each separately
                    if self.dry_run:
                        print(f"Would tokenize directory with subdirectories: {local_input_path}")
                        for subdir in subdirs:
                            sub_dest_path = self.local_dir / self.output_prefix / relative_path / subdir.name / self.tokenizer.replace('/', '/')
                            # Test ID field detection even in dry run for better preview
                            subdir_relative_path = str(subdir.relative_to(self.local_dir))
                            sub_id_field_name, sub_id_field_type = self._detect_id_field_issues(subdir_relative_path)
                            print(f"  Subdir: {subdir.name} -> {sub_dest_path}")
                            if sub_id_field_name:
                                print(f"    ID field: {sub_id_field_name} ({sub_id_field_type})")
                    else:
                        for subdir in subdirs:
                            sub_doc_path = f"{subdir}/*"
                            sub_dest_path = self.local_dir / self.output_prefix / relative_path / subdir.name / self.tokenizer.replace('/', '/')
                            
                            # Detect ID field issues for this specific subdirectory
                            # Pass the relative path from local_dir to the subdirectory
                            subdir_relative_path = str(subdir.relative_to(self.local_dir))
                            sub_id_field_name, sub_id_field_type = self._detect_id_field_issues(subdir_relative_path)
                            
                            success = self._run_tokenization_command(
                                sub_doc_path, sub_dest_path, tokenizer_dir, 
                                sub_id_field_name, sub_id_field_type
                            )
                            if not success:
                                errors.append(f"Failed to tokenize {relative_path}/{subdir.name}")
                    continue
                else:
                    # Empty directory or no subdirs, treat as regular
                    doc_path = f"{local_input_path}/*"
                    dest_path = self.local_dir / self.output_prefix / relative_path / self.tokenizer.replace('/', '/')
                    
                    if self.dry_run:
                        print(f"Would tokenize empty/regular directory:")
                        print(f"  Input:  {doc_path}")
                        print(f"  Output: {dest_path}")
            
            if not self.dry_run:
                success = self._run_tokenization_command(
                    doc_path, dest_path, tokenizer_dir, id_field_name, id_field_type
                )
                if not success:
                    errors.append(f"Failed to tokenize {relative_path}")
        
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
            "dolma", "tokens",
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
        if id_field_name is not None and id_field_name != 'id':
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
        """Upload tokenized data to remote storage (both S3 and GCS)."""
        print("=== UPLOADING DATA ===")
        if self.dry_run:
            print("DRY RUN MODE: Showing what would be uploaded")
        
        errors = []
        
        for path in self.paths:
            # Calculate relative path by removing input prefix
            if path.startswith(self.input_prefix):
                relative_path = path[len(self.input_prefix):].lstrip('/')
            else:
                relative_path = path
            
            # Find tokenized directory based on new structure
            if path.endswith(('.gz', '.jsonl', '.json')):
                # Single file case
                dest_base = relative_path
                for ext in ['.jsonl.gz', '.json.gz', '.gz', '.jsonl', '.json']:
                    if dest_base.endswith(ext):
                        dest_base = dest_base[:-len(ext)]
                        break
                local_tokenized_path = self.local_dir / self.output_prefix / dest_base / self.tokenizer.replace('/', '/')
                s3_remote_dest = f"{self.remote_prefix}{self.output_prefix.rstrip('/')}/{dest_base}/{self.tokenizer}/"
                gcs_remote_dest = f"gs://{self.output_prefix.rstrip('/')}/{dest_base}/{self.tokenizer}/"
                
                if self.dry_run:
                    print(f"Would upload file tokenization:")
                    print(f"  Local:  {local_tokenized_path}")
                    print(f"  S3:     {s3_remote_dest}")
                    print(f"  GCS:    {gcs_remote_dest}")
                
            else:
                # Directory case - check if it has subdirectories that were processed separately
                local_input_path = self.local_dir / path
                has_subdirs = False
                subdirs = []
                
                # Always check filesystem for proper detection (even in dry run)
                if local_input_path.exists():
                    subdirs = [d for d in local_input_path.iterdir() if d.is_dir()]
                    has_immediate_gz = bool(list(local_input_path.glob("*.gz")))
                    has_subdirs = subdirs and not has_immediate_gz
                
                if has_subdirs:
                    # Upload each subdirectory separately
                    if self.dry_run:
                        print(f"Would upload directory with subdirectories: {path}")
                        for subdir in subdirs:
                            sub_local_path = self.local_dir / self.output_prefix / relative_path / subdir.name / self.tokenizer.replace('/', '/')
                            sub_s3_remote_dest = f"{self.remote_prefix}{self.output_prefix.rstrip('/')}/{relative_path.rstrip('/')}/{subdir.name}/{self.tokenizer}/"
                            sub_gcs_remote_dest = f"gs://{self.output_prefix.rstrip('/')}/{relative_path.rstrip('/')}/{subdir.name}/{self.tokenizer}/"
                            print(f"  Subdir: {sub_local_path}")
                            print(f"    S3:  {sub_s3_remote_dest}")
                            print(f"    GCS: {sub_gcs_remote_dest}")
                    else:
                        for subdir in subdirs:
                            sub_local_path = self.local_dir / self.output_prefix / relative_path / subdir.name / self.tokenizer.replace('/', '/')
                            sub_s3_remote_dest = f"{self.remote_prefix}{self.output_prefix.rstrip('/')}/{relative_path.rstrip('/')}/{subdir.name}/{self.tokenizer}/"
                            sub_gcs_remote_dest = f"gs://{self.output_prefix.rstrip('/')}/{relative_path.rstrip('/')}/{subdir.name}/{self.tokenizer}/"
                            
                            if not sub_local_path.exists():
                                errors.append(f"Tokenized data not found: {sub_local_path}")
                                continue
                            
                            # Upload to S3
                            s3_cmd = ["s5cmd", "sync", f"{sub_local_path}/", sub_s3_remote_dest]
                            success, output = self._run_command(s3_cmd, f"Uploading {sub_local_path} to S3")
                            
                            if not success:
                                errors.append(f"Failed to upload {sub_local_path} to S3: {output}")
                            else:
                                print(f"Successfully uploaded to S3: {sub_s3_remote_dest}")
                            
                            # Upload to GCS - try gcloud storage first, fallback to gsutil
                            # gcloud storage rsync often handles auth better than gsutil
                            gcs_cmd = ["gcloud", "storage", "rsync", "-r", f"{sub_local_path}/", sub_gcs_remote_dest]
                            success, output = self._run_command(gcs_cmd, f"Uploading {sub_local_path} to GCS (using gcloud storage)")
                            
                            if not success:
                                print("Retrying with gsutil...")
                                gcs_cmd = ["gsutil", "-m", "rsync", "-r", f"{sub_local_path}/", sub_gcs_remote_dest]
                                success, output = self._run_command(gcs_cmd, f"Uploading {sub_local_path} to GCS (using gsutil)")
                            
                            if not success:
                                errors.append(f"Failed to upload {sub_local_path} to GCS: {output}")
                            else:
                                print(f"Successfully uploaded to GCS: {sub_gcs_remote_dest}")
                    continue
                else:
                    # Regular directory
                    local_tokenized_path = self.local_dir / self.output_prefix / relative_path / self.tokenizer.replace('/', '/')
                    s3_remote_dest = f"{self.remote_prefix}{self.output_prefix.rstrip('/')}/{relative_path.rstrip('/')}/{self.tokenizer}/"
                    gcs_remote_dest = f"gs://{self.output_prefix.rstrip('/')}/{relative_path.rstrip('/')}/{self.tokenizer}/"
                    
                    if self.dry_run:
                        print(f"Would upload regular directory:")
                        print(f"  Local:  {local_tokenized_path}")
                        print(f"  S3:     {s3_remote_dest}")
                        print(f"  GCS:    {gcs_remote_dest}")
            
            if self.dry_run:
                continue
            
            if not local_tokenized_path.exists():
                errors.append(f"Tokenized data not found: {local_tokenized_path}")
                continue
            
            # Upload to S3
            s3_cmd = ["s5cmd", "sync", f"{local_tokenized_path}/", s3_remote_dest]
            success, output = self._run_command(s3_cmd, f"Uploading {local_tokenized_path} to S3")
            
            if not success:
                errors.append(f"Failed to upload {local_tokenized_path} to S3: {output}")
            else:
                print(f"Successfully uploaded to S3: {s3_remote_dest}")
            
            # Upload to GCS - try gcloud storage first, fallback to gsutil
            # gcloud storage rsync often handles auth better than gsutil
            gcs_cmd = ["gcloud", "storage", "rsync", "-r", f"{local_tokenized_path}/", gcs_remote_dest]
            success, output = self._run_command(gcs_cmd, f"Uploading {local_tokenized_path} to GCS (using gcloud storage)")
            
            if not success:
                print("Retrying with gsutil...")
                gcs_cmd = ["gsutil", "-m", "rsync", "-r", f"{local_tokenized_path}/", gcs_remote_dest]
                success, output = self._run_command(gcs_cmd, f"Uploading {local_tokenized_path} to GCS (using gsutil)")
            
            if not success:
                errors.append(f"Failed to upload {local_tokenized_path} to GCS: {output}")
            else:
                print(f"Successfully uploaded to GCS: {gcs_remote_dest}")
        
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
        
        errors = []
        
        for path in self.paths:
            # Determine relative path from input prefix
            if path.startswith(self.input_prefix):
                relative_path = path[len(self.input_prefix):].lstrip('/')
            else:
                relative_path = path
            
            # Check if the local data exists (required for destination_paths)
            local_path = self.local_dir / path
            if not local_path.exists():
                errors.append(f"Local data not found: {local_path}")
                continue
            
            if path.endswith(('.gz', '.jsonl', '.json')):
                # Single file case - check if tokenized output exists
                dest_base = relative_path
                for ext in ['.jsonl.gz', '.json.gz', '.gz', '.jsonl', '.json']:
                    if dest_base.endswith(ext):
                        dest_base = dest_base[:-len(ext)]
                        break
                
                # Check if tokenized directory exists
                tokenized_path = self.local_dir / self.output_prefix / dest_base / self.tokenizer.replace('/', '/')
                if not tokenized_path.exists():
                    errors.append(f"Tokenized data not found: {tokenized_path}")
                    continue
                
                base_dest = f"{self.remote_prefix}{self.output_prefix.rstrip('/')}/{dest_base}/{self.tokenizer}"
                wildcard_path = f"{base_dest}/*.npy"
                print(wildcard_path)
                
            else:
                # Directory case - check if it has subdirectories or immediate files
                subdirs = [d for d in local_path.iterdir() if d.is_dir()]
                has_immediate_gz = bool(list(local_path.glob("*.gz")))
                has_subdirs = subdirs and not has_immediate_gz
                
                if has_subdirs:
                    # Directory with subdirectories - check if at least one tokenized subdir exists
                    found_tokenized = False
                    for subdir in subdirs:
                        tokenized_subdir = self.local_dir / self.output_prefix / relative_path / subdir.name / self.tokenizer.replace('/', '/')
                        if tokenized_subdir.exists():
                            found_tokenized = True
                            break
                    
                    if not found_tokenized:
                        errors.append(f"No tokenized subdirectories found for: {local_path}")
                        continue
                    
                    # Use wildcard for all subdirs
                    base_dest = f"{self.remote_prefix}{self.output_prefix.rstrip('/')}/{relative_path.rstrip('/')}/*/{self.tokenizer}"
                    wildcard_path = f"{base_dest}/*.npy"
                    print(wildcard_path)
                    
                else:
                    # Regular directory - check if tokenized directory exists
                    tokenized_path = self.local_dir / self.output_prefix / relative_path / self.tokenizer.replace('/', '/')
                    if not tokenized_path.exists():
                        errors.append(f"Tokenized data not found: {tokenized_path}")
                        continue
                    
                    base_dest = f"{self.remote_prefix}{self.output_prefix.rstrip('/')}/{relative_path.rstrip('/')}/{self.tokenizer}"
                    wildcard_path = f"{base_dest}/*.npy"
                    print(wildcard_path)
        
        if errors:
            print("\n" + "="*60)
            print("DESTINATION PATHS ERRORS")
            print("="*60)
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
            print(f"\n{len(errors)} path(s) missing tokenized data out of {len(self.paths)} total paths.")
            print("ACTION REQUIRED: Run download and tokenize commands first to generate the required data.")
            sys.exit(1)


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
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually executing commands")
    
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
        tokenizer=args.tokenizer,
        dry_run=args.dry_run
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
