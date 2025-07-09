#!/bin/bash

prefix="ai2-llm/pretraining-data/sources/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2_denylisted_reshard_length-buckets_sample10pct"


if [ ! -d "/mnt/raid0/${prefix}" ]; then
    s5cmd cp -sp "s3://${prefix}/*" "/mnt/raid0/${prefix}/"
fi

tokenizer_name="allenai/dolma2-tokenizer"

uv run huggingface-cli download "${tokenizer_name}" --local-dir /mnt/raid0/tokenizer


for length in $(ls /mnt/raid0/${prefix})
do
    echo "Processing $length"
    dest="$(echo "$prefix" | sed 's|/pretraining-data/sources/|/preprocessed/|g')/${tokenizer_name}/${length}/"

    uv run dolma tokens \
    --documents "/mnt/raid0/${prefix}/${length}/*/*.jsonl.zst" \
    --destination "/mnt/raid0/${dest}" \
    --tokenizer.name_or_path "/mnt/raid0/tokenizer/tokenizer.json"\
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --ring_size 8 \
    --processes $(( $(nproc) - 2 )) \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype 'uint32'
done
