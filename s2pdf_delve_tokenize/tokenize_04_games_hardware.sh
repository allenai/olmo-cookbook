#!/bin/bash
set -e

# Processing script for categories: games and hardware

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
# Process games
####################
echo "========================================="
echo "Processing category: games"
echo "========================================="

LOCAL_BASE="/mnt/raid0/s2pdf_delve_games"
OUT_DIR="${LOCAL_BASE}"
DST_S3="s3://ai2-llm/preprocessed/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2/games"

# Create local directory
echo "Creating local directory for games..."
mkdir -p "$OUT_DIR"

# Copy data from S3
echo "Copying games from S3..."
s5cmd cp --concurrency 128 "$S3_SOURCE/games/*" "$OUT_DIR/"

# Run tokenization
echo "Running tokenization for games..."
uv run dolma tokens \
    --documents "$OUT_DIR/step_final/*.gz" \
    --destination "$DST_S3" \
    --tokenizer.name_or_path "$TOKENIZER_PATH" \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --ring_size 16 \
    --processes "$(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count() - 4)')" \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype 'uint32' \
    --work_dir.input /mnt/raid0/tmpin \
    --work_dir.output /mnt/raid0/tmpout


echo "Completed processing games"

####################
# Process hardware
####################
echo "========================================="
echo "Processing category: hardware"
echo "========================================="

LOCAL_BASE="/mnt/raid0/s2pdf_delve_hardware"
OUT_DIR="${LOCAL_BASE}"
DST_S3="s3://ai2-llm/preprocessed/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2/hardware"

# Create local directory
echo "Creating local directory for hardware..."
mkdir -p "$OUT_DIR"

# Copy data from S3
echo "Copying hardware from S3..."
s5cmd cp --concurrency 128 "$S3_SOURCE/hardware/*" "$OUT_DIR/"

# Run tokenization
echo "Running tokenization for hardware..."
uv run dolma tokens \
    --documents "$OUT_DIR/step_final/*.gz" \
    --destination "$DST_S3" \
    --tokenizer.name_or_path "$TOKENIZER_PATH" \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --ring_size 16 \
    --processes "$(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count() - 4)')" \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype 'uint32' \
    --work_dir.input /mnt/raid0/tmpin \
    --work_dir.output /mnt/raid0/tmpout


echo "Completed processing hardware"
echo "========================================="
echo "Completed all processing for games and hardware"
echo "========================================="
