#!/usr/bin/env bash

models=(
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-all-dressed-v18-ce7c0c2a/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-all-dressed-v20-1e29004d/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-hq-web-baseline-0e513fe6/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-actual-5d35521f/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-distill-a2d7cd25/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-extract-knowledge-76f1ff02/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-knowledge_list-de0e6931/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-qa-d2535212/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-wrap_medium-ab332b66/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-olmo3-mix-2b9cd813/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-round1-olmo3_7b-with-reasoning-anneal-03fb7a8d/step4769'
    'ai2-llm/checkpoints/mayeec/olmo3-microanneal-round1-olmo3_7b-no-reasoning-anneal-25249a74/step4769'
)


# Moving checkpoints to weka
for model in "${models[@]}"; do
    uv run python -m cookbook.remote gs://$model weka://oe-training-default/$model
done


# Convert checkpoints
for model in "${models[@]}"; do
    uv run olmo-cookbook-eval convert \
        "/oe-training-default/$model" \
        -t olmo-core-v2 \
        --use-beaker \
        --olmo-core-v2-commit-hash  013ef7b54aa2d583f9811ec6211a536da407a4b1 \
        --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
        --huggingface-transformers-commit-hash ca728b8879ce5127ea3e2f8d309c2c5febab5dc5
done


# Launch regular evals
for model in "${models[@]}"; do
    uv run olmo-cookbook-eval evaluate \
        "/oe-training-default/${model}-hf" \
        --tasks olmo3:dev:7b:main \
        --priority high \
        --cluster aus80g \
        --partition-size 8 \
        --num-gpus 1 \
        --model-backend vllm \
        --model-args trust_remote_code=true,max_length=4096 \
        --beaker-image oe-eval-beaker/oe_eval_qk_norm_auto \
        --dashboard olmo3-midtraining-web \
        --workspace ai2/oe-data
done


# Launch reasoning evals
for model in "${models[@]}"; do
    uv run olmo-cookbook-eval evaluate \
        "/oe-training-default/${model}-hf" \
        --tasks olmo3:dev:midtrain:v0 \
        --priority high \
        --cluster aus80g \
        --num-gpus 1 \
        --partition-size 8 \
        --model-backend vllm \
        --no-compute-gold-bpb \
        --model-args chat_template=basic_answer,trust_remote_code=true,max_length=8192 \
        --use-gantry \
        --gantry-args env-secret="OPENAI_API_KEY=openai_api_key" \
        --task-args chat_overrides="{\"generation_kwargs\": {\"stop_sequences\": [\"Problem:\", \"Answer:\", \"Question:\", \"</s>\", \"<|eot_id|>\"]}}" \
        --oe-eval-branch davidh/head-qk-norm \
        --beaker-image oe-eval-beaker/oe_eval_qk_norm_auto \
        --dashboard olmo3-midtraining-web \
        --workspace ai2/oe-data
done
