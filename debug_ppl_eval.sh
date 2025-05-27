#!/bin/bash



olmo-cookbook-eval evaluate "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0000/step22100-hf" \
    --tasks ultrachat_masked_ppl \
    --compute-gold-bpb \
    --priority high \
    --cluster 80g \
    --num-gpus 1 \
    --model-backend hf \
    --model-args dtype=bfloat16 \
    --partition-size 8 \
    --dashboard regmixer  \
    --workspace ai2/oe-data \
    --gantry-args '{"hf_token": "MAYEEC_HF_TOKEN"}'
