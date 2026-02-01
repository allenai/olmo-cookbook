#!/bin/bash

MODEL_PATH="gs://ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T_M100B_r5-midtrain_round3_qwenlike_pre_s2pdf_gzip2080_cweN-yake-all-olmo_fullonly_10B-80b43be5/step2385"
DASHBOARD=stego32
WORKSPACE=ai2/Olmo_3
LENGTHS=(4)

for length in "${LENGTHS[@]}"; do
  uv run --python=3.12 --extra=all \
    olmo-cookbook-eval evaluate \
    $MODEL_PATH \
    --priority high \
    --cluster ai2/augusta-google-1 \
    --num-gpus 1 \
    --model-backend vllm \
    --dashboard $DASHBOARD \
    --budget ai2/oe-base \
    --vllm-use-v1-spec \
    --model-args 'trust_remote_code=true, chat_model=null, max_length=65536' \
    --task-args 'use_chat_format=false' \
    --gantry-args='install=echo' \
    --workspace $WORKSPACE \
    --partition-size 4 \
    --tasks "ruler:${length}k"
done
