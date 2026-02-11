#!/usr/bin/env bash


set -e

# Run once on all nodes manually
# uv run huggingface-cli download allenai/dolma2-tokenizer --local-dir /mnt/raid0/tokenizers/allenai/dolma2-tokenizer 
# s5cmd cp s3://ai2-llm/pretraining-data/sources/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2_denylisted_reshard_denyagain_compressionv2/* /mnt/raid0/ai2-llm/pretraining-data/sources/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2_denylisted_reshard_denyagain_compressionv2


src_path="/mnt/raid0/ai2-llm/pretraining-data/sources/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2_denylisted_reshard_denyagain_compressionv2/"
tokenizer_name="allenai/dolma2-tokenizer"
total_cores=$(nproc)

output_dir="/mnt/raid0/allenai/dolma2-tokenizer/crime_law" 

mkdir -p "$output_dir"

uv run dolma tokens \
    --documents "$src_path/crime_law/*.zst" \
    --destination "$output_dir" \
    --tokenizer.name_or_path /mnt/raid0/tokenizers/allenai/dolma2-tokenizer/tokenizer.json \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --processes $total_cores \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype 'uint32' 

s5cmd cp -sp "$output_dir/" s3://ai2-llm/preprocessed/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2_denylisted_reshard_denyagain_compressionv2/crime_law/allenai/dolma2-tokenizer/