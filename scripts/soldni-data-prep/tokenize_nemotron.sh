#!/bin/bash


prefixes=(
    "/mnt/raid0/ai2-llm/pretraining-data/sources/Nemotron-CC/v0/quality=high/kind=actual/kind2=actual"
    "/mnt/raid0/ai2-llm/pretraining-data/sources/Nemotron-CC/v0/quality=high/kind=synthetic/kind2=distill"
    "/mnt/raid0/ai2-llm/pretraining-data/sources/Nemotron-CC/v0/quality=high/kind=synthetic/kind2=diverse_qa_pairs"
    "/mnt/raid0/ai2-llm/pretraining-data/sources/Nemotron-CC/v0/quality=high/kind=synthetic/kind2=extract_knowledge"
    "/mnt/raid0/ai2-llm/pretraining-data/sources/Nemotron-CC/v0/quality=high/kind=synthetic/kind2=knowledge_list"
    "/mnt/raid0/ai2-llm/pretraining-data/sources/Nemotron-CC/v0/quality=high/kind=synthetic/kind2=wrap_medium"
)


uv run huggingface-cli download allenai/dolma2-tokenizer --local-dir /mnt/raid0/tokenizer


for prefix in "${prefixes[@]}"
do
    echo "Processing $prefix"
    dest=$(echo "$prefix" | sed 's|/pretraining-data/sources/|/preprocessed/|g' | sed 's|/v0/|/v0/allenai/dolma2-tokenizer/|g')

    uv run dolma tokens \
    --documents "$prefix/*.jsonl.zstd" \
    --destination "$dest" \
    --tokenizer.name_or_path "/mnt/raid0/tokenizer/tokenizer.json"\
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --fields.id_field_name warc_record_id \
    --ring_size 8 \
    --processes 64 \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype 'uint32'
done
