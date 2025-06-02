#!/usr/bin/env python3
import os
from pathlib import Path

CATEGORIES = [
    "adult",
    "art_design",
    "crime_law",
    "education_jobs",
    "entertainment",
    "fashion_beauty",
    "finance_business",
    "food_dining",
    "games",
    "hardware",
    "health",
    "history",
    "home_hobbies",
    "industrial",
    "literature",
    "politics",
    "religion",
    "science_tech",
    "social_life",
    "software",
    "software_dev",
    "sports_fitness",
    "transportation",
    "travel",
]

SCRIPT_TEMPLATE = """#!/bin/bash
set -e

# Processing script for categories: {cat1} and {cat2}

# Set common variables
S3_SOURCE="s3://ai2-llm/pretraining-data/sources/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2"
TOKENIZER_DIR="/mnt/raid0/dolma2-tokenizer"
TOKENIZER_PATH="$TOKENIZER_DIR/tokenizer.json"

# Download tokenizer from Hugging Face if not already present
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "Downloading dolma2-tokenizer from Hugging Face..."
    mkdir -p "$TOKENIZER_DIR"
    uv run huggingface-cli download allenai/dolma2-tokenizer --local-dir "$TOKENIZER_DIR"
    echo "Tokenizer downloaded successfully!"
else
    echo "Tokenizer already exists at $TOKENIZER_PATH"
fi

####################
# Process {cat1}
####################
echo "========================================="
echo "Processing category: {cat1}"
echo "========================================="

LOCAL_BASE="/mnt/raid0/s2pdf_delve_{cat1}"
OUT_DIR="${{LOCAL_BASE}}"
DST_S3="s3://ai2-llm/preprocessed/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2/{cat1}"

# Create local directory
echo "Creating local directory for {cat1}..."
mkdir -p "$OUT_DIR"

# Copy data from S3
echo "Copying {cat1} from S3..."
s5cmd cp --concurrency 128 "$S3_SOURCE/{cat1}/*" "$OUT_DIR/"

# Run tokenization
echo "Running tokenization for {cat1}..."
uv run dolma tokens \\
    --documents "$OUT_DIR/step_final/*.gz" \\
    --destination "$DST_S3" \\
    --tokenizer.name_or_path "$TOKENIZER_PATH" \\
    --tokenizer.eos_token_id 100257 \\
    --tokenizer.pad_token_id 100277 \\
    --tokenizer.segment_before_tokenization \\
    --tokenizer.encode_special_tokens \\
    --ring_size 16 \\
    --processes "$(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count() - 4)')" \\
    --max_size 4_000_000_000 \\
    --sample_ring_prop \\
    --dtype 'uint32' \\
    --work_dir.input /mnt/raid0/tmpin \\
    --work_dir.output /mnt/raid0/tmpout


echo "Completed processing {cat1}"

####################
# Process {cat2}
####################
echo "========================================="
echo "Processing category: {cat2}"
echo "========================================="

LOCAL_BASE="/mnt/raid0/s2pdf_delve_{cat2}"
OUT_DIR="${{LOCAL_BASE}}"
DST_S3="s3://ai2-llm/preprocessed/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2/{cat2}"

# Create local directory
echo "Creating local directory for {cat2}..."
mkdir -p "$OUT_DIR"

# Copy data from S3
echo "Copying {cat2} from S3..."
s5cmd cp --concurrency 128 "$S3_SOURCE/{cat2}/*" "$OUT_DIR/"

# Run tokenization
echo "Running tokenization for {cat2}..."
uv run dolma tokens \\
    --documents "$OUT_DIR/step_final/*.gz" \\
    --destination "$DST_S3" \\
    --tokenizer.name_or_path "$TOKENIZER_PATH" \\
    --tokenizer.eos_token_id 100257 \\
    --tokenizer.pad_token_id 100277 \\
    --tokenizer.segment_before_tokenization \\
    --tokenizer.encode_special_tokens \\
    --ring_size 16 \\
    --processes "$(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count() - 4)')" \\
    --max_size 4_000_000_000 \\
    --sample_ring_prop \\
    --dtype 'uint32' \\
    --work_dir.input /mnt/raid0/tmpin \\
    --work_dir.output /mnt/raid0/tmpout


echo "Completed processing {cat2}"
echo "========================================="
echo "Completed all processing for {cat1} and {cat2}"
echo "========================================="
"""

def main():
    script_dir = Path(__file__).parent
    
    # Generate consecutive pairs of categories (0-1, 2-3, 4-5, etc.)
    category_pairs = []
    for i in range(0, len(CATEGORIES), 2):
        if i + 1 < len(CATEGORIES):
            category_pairs.append((CATEGORIES[i], CATEGORIES[i + 1]))
    
    print(f"Generating {len(category_pairs)} shell scripts for consecutive category pairs...")
    
    for i, (cat1, cat2) in enumerate(category_pairs):
        script_name = f"tokenize_{i:02d}_{cat1}_{cat2}.sh"
        script_path = script_dir / script_name
        
        script_content = SCRIPT_TEMPLATE.format(cat1=cat1, cat2=cat2)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"Created: {script_name} (categories {i*2}-{i*2+1}: {cat1}, {cat2})")
    
    print(f"\nTotal scripts created: {len(category_pairs)}")
    print(f"Scripts saved in: {script_dir}")

if __name__ == "__main__":
    main()