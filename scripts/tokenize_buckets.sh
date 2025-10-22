#!/bin/bash

set -e

# Default values
DRY_RUN=false
SEED=42
DTYPE="uint32"
TOKENIZER_NAME="allenai/dolma2-tokenizer"
BOS_TOKEN_ID="null"
EOS_TOKEN_ID=100257
PAD_TOKEN_ID=100277
IGNORE_CATEGORIES=""
SPECIFIC_CATEGORY=""
SPECIFIC_BUCKETS=""
USE_BLOB_ID=false
MAX_CORES=128

# Function to show usage
usage() {
  echo "Usage: $0 --input-dir <input_directory> --output-dir <output_directory> [options]"
  echo "  --input-dir: Directory containing category subdirectories"
  echo "  --output-dir: Directory where tokenized output will be saved"
  echo "  --dry-run: Show commands without executing them"
  echo "  --ignore-categories: Comma-separated list of categories to ignore"
  echo "  --category: Process only this specific category"
  echo "  --buckets: Comma-separated list of specific buckets to process"
  echo "  --use-blob-id: Use blob_id as the id field name"
  echo "  --max-cores: Maximum number of cores to use concurrently (default: 128)"
  echo "  -h, --help: Show this help message"
  exit 1
}

# Function to check if category should be ignored
should_ignore_category() {
  local category="$1"
  if [[ -n "$IGNORE_CATEGORIES" ]]; then
    IFS=',' read -ra IGNORE_ARRAY <<< "$IGNORE_CATEGORIES"
    for ignore_cat in "${IGNORE_ARRAY[@]}"; do
      if [[ "$category" == "$ignore_cat" ]]; then
        return 0  # true - should ignore
      fi
    done
  fi
  return 1  # false - should not ignore
}

# Function to check if bucket should be processed
should_process_bucket() {
  local bucket="$1"
  if [[ -n "$SPECIFIC_BUCKETS" ]]; then
    IFS=',' read -ra BUCKET_ARRAY <<< "$SPECIFIC_BUCKETS"
    for specific_bucket in "${BUCKET_ARRAY[@]}"; do
      if [[ "$bucket" == "$specific_bucket" ]]; then
        return 0  # true - should process
      fi
    done
    return 1  # false - not in specific buckets list
  fi
  return 0  # true - no specific buckets filter, process all
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --ignore-categories)
      IGNORE_CATEGORIES="$2"
      shift 2
      ;;
    --category)
      SPECIFIC_CATEGORY="$2"
      shift 2
      ;;
    --buckets)
      SPECIFIC_BUCKETS="$2"
      shift 2
      ;;
    --use-blob-id)
      USE_BLOB_ID=true
      shift
      ;;
    --max-cores)
      MAX_CORES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Check required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Error: Both --input-dir and --output-dir are required"
  usage
fi

# Check if both ignore and specific category are set
if [[ -n "$IGNORE_CATEGORIES" && -n "$SPECIFIC_CATEGORY" ]]; then
  echo "Error: Cannot use both --ignore-categories and --category options together"
  exit 1
fi

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: Input directory '$INPUT_DIR' does not exist"
  exit 1
fi

# If specific category is set, check if it exists
if [[ -n "$SPECIFIC_CATEGORY" && ! -d "$INPUT_DIR/$SPECIFIC_CATEGORY" ]]; then
  echo "Error: Specific category '$SPECIFIC_CATEGORY' does not exist in input directory"
  exit 1
fi

# Create output directory if it doesn't exist (unless dry run)
if [[ "$DRY_RUN" == false && ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Check if GNU parallel is available
if ! command -v parallel &> /dev/null; then
  echo "Error: GNU parallel is not installed. Please install it with: sudo apt-get install parallel"
  exit 1
fi

# Arrays to store commands and their core requirements
declare -a COMMANDS=()
declare -a CORE_COUNTS=()

# Function to add command to array with core count
add_command() {
  local cmd="$1"
  local cores="$2"
  COMMANDS+=("$cmd")
  CORE_COUNTS+=("$cores")
}

# Build the command with optional blob_id field and return both command and core count
build_command() {
  local subdir_path="$1"
  local relative_path="$2"

  # Count the number of .jsonl.zst files in the subdirectory
  local file_count=$(ls "$subdir_path"/*.jsonl.zst 2>/dev/null | wc -l)

  # Set processes to the number of files or 8 as minimum
  local processes=$(( file_count > 8 ? file_count : 8 ))

  local cmd="uv run dolma tokens --destination \"$OUTPUT_DIR/$relative_path\" --documents \"$subdir_path/*.jsonl.zst\" --processes $processes --seed $SEED --dtype $DTYPE --tokenizer.name_or_path $TOKENIZER_NAME --tokenizer.eos_token_id $EOS_TOKEN_ID --tokenizer.pad_token_id $PAD_TOKEN_ID"

  if [[ "$USE_BLOB_ID" == true ]]; then
    cmd="$cmd --fields.id_field_name blob_id"
  fi

  # Return command and process count separated by a delimiter
  echo "$cmd|$processes"
}

# Function to recursively collect subdirectories
collect_subdirectories() {
  local base_path="$1"
  local category="$2"

  # Find all subdirectories that contain .jsonl.zst files
  # Use process substitution to avoid subshell issues
  while IFS= read -r -d '' subdir; do
    # Skip the base category directory itself
    if [[ "$subdir" == "$base_path" ]]; then
      continue
    fi

    # Extract bucket name from the subdirectory path
    bucket_name=$(basename "$subdir")

    # Check if this bucket should be processed
    if ! should_process_bucket "$bucket_name"; then
      if [[ "$DRY_RUN" == true ]]; then
        echo "Skipping bucket: $bucket_name (not in specified buckets list)"
      fi
      continue
    fi

    # Check if this subdirectory has .jsonl.zst files
    if ls "$subdir"/*.jsonl.zst >/dev/null 2>&1; then
      relative_subdir=$(echo "$subdir" | sed "s|^$INPUT_DIR/||")
      cmd_info=$(build_command "$subdir" "$relative_subdir")

      # Split command and core count
      cmd="${cmd_info%|*}"
      cores="${cmd_info##*|}"

      if [[ "$DRY_RUN" == true ]]; then
        echo "Would run ($cores cores): $cmd"
      else
        echo "Collecting command for subdirectory: $relative_subdir (requires $cores cores)"
        add_command "$cmd" "$cores"
      fi
    fi
  done < <(find "$base_path" -type d -print0)
}

# Function to execute all commands in parallel with core management
execute_parallel() {
  if [[ ${#COMMANDS[@]} -eq 0 ]]; then
    echo "No commands to execute"
    return 0
  fi

  local total_cores=0
  for cores in "${CORE_COUNTS[@]}"; do
    total_cores=$((total_cores + cores))
  done

  echo "Executing ${#COMMANDS[@]} commands requiring $total_cores total cores with up to $MAX_CORES concurrent cores..."

  # Simple bin packing: calculate how many jobs can run concurrently
  # Sort commands by core count (descending) for better packing
  local temp_dir=$(mktemp -d)
  local sorted_file="$temp_dir/sorted_jobs.txt"

  # Create job file with core count and command
  for i in "${!COMMANDS[@]}"; do
    echo "${CORE_COUNTS[$i]}|${COMMANDS[$i]}" >> "$sorted_file"
  done

  # Sort by core count (descending) for better bin packing
  sort -t'|' -k1,1nr "$sorted_file" > "$sorted_file.tmp"
  mv "$sorted_file.tmp" "$sorted_file"

  # Execute with core-aware parallelization using GNU parallel semaphore
  local joblog_file=$(mktemp)

  # Use a simpler approach: run jobs sequentially in batches that fit within MAX_CORES
  # Create command batches that don't exceed MAX_CORES
  local batch_file="$temp_dir/current_batch.txt"
  local batch_cores=0
  local batch_num=1

  > "$batch_file"  # Clear batch file

  echo "Creating core-aware batches..."

  while IFS='|' read -r cores cmd; do
    # If adding this job would exceed MAX_CORES, execute current batch
    if (( batch_cores + cores > MAX_CORES && batch_cores > 0 )); then
      echo "Executing batch $batch_num with $batch_cores cores..."

      # Execute current batch in parallel
      if parallel --will-cite --halt never --joblog "$joblog_file" --jobs 0 < "$batch_file"; then
        echo "Batch $batch_num completed"
      else
        echo "Batch $batch_num completed with some failures"
      fi

      # Start new batch
      > "$batch_file"
      batch_cores=0
      batch_num=$((batch_num + 1))
    fi

    # Add job to current batch
    echo "$cmd" >> "$batch_file"
    batch_cores=$((batch_cores + cores))

  done < "$sorted_file"

  # Execute final batch if it has any jobs
  if [[ -s "$batch_file" ]]; then
    echo "Executing final batch $batch_num with $batch_cores cores..."
    if parallel --will-cite --halt never --joblog "$joblog_file" --jobs 0 < "$batch_file"; then
      echo "Final batch completed"
    else
      echo "Final batch completed with some failures"
    fi
  fi

  # Check for failures in the job log
  local failed_jobs=$(awk 'NR>1 && $7!=0 {print $9}' "$joblog_file")
  if [[ -n "$failed_jobs" ]]; then
    echo "The following commands failed:"
    echo "$failed_jobs"
    echo "Check the output above for details."
  else
    echo "All commands completed successfully"
  fi

  rm -rf "$temp_dir" "$joblog_file"
  return 0
}

# Collect all commands first
if [[ -n "$SPECIFIC_CATEGORY" ]]; then
  # Process only the specific category
  category_path="$INPUT_DIR/$SPECIFIC_CATEGORY"
  category="$SPECIFIC_CATEGORY"

  # Skip if category should be ignored
  if should_ignore_category "$category"; then
    echo "Skipping ignored category: $category"
  else
    echo "Processing category: $category"
    collect_subdirectories "$category_path" "$category"
  fi
else
  # Process all categories (except ignored ones)
  for category_path in "$INPUT_DIR"/*; do
    if [[ -d "$category_path" ]]; then
      category=$(basename "$category_path")

      # Skip if category should be ignored
      if should_ignore_category "$category"; then
        echo "Skipping ignored category: $category"
        continue
      fi

      echo "Processing category: $category"
      collect_subdirectories "$category_path" "$category"
    fi
  done
fi

# Execute all commands in parallel (unless dry run)
if [[ "$DRY_RUN" == false ]]; then
  execute_parallel
fi

echo "Done!"
