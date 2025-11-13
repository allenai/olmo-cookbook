#!/usr/bin/env bash
set -euo pipefail

# paths=(
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step2000-tokens17B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step4000-tokens34B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step6000-tokens51B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step9000-tokens76B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step18000-tokens151B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step36000-tokens302B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step72000-tokens604B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step145000-tokens1217B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step289000-tokens2425B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step467000-tokens3918B"
#     "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-32b/stage1-step721901-tokens6056B"
# )
# paths=(
#   "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step20000-tokens42B"
#   "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step100000-tokens210B"
#   "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step190000-tokens399B"
#   "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step380000-tokens797B"
#   "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step760000-tokens1594B"
#   "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step1140000-tokens2391B"
#   "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step1530000-tokens3209B"
#   "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step1907359-tokens4001B"
# )
paths=(
  # "allenai/OLMo-2-0425-1B-Instruct"
  # "allenai/OLMo-2-1124-7B-Instruct"
  # "allenai/OLMo-2-1124-13B-Instruct"
  # "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage1-step1907359-tokens4001B"
  # "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-1b/stage2-ingredient3-step23852-tokens51B"
  # "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-7b/stage1-step928646-tokens3896B"
  # "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-7b/stage2-ingredient3-step11931-tokens50B"
  # "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-13b/stage1-step596057-tokens5001B"
  # "/oe-training-default/ai2-llm/checkpoints/yapeic/olmo2-13b/stage2-ingredient4-step35773-tokens300B"
)

for model_path in "${paths[@]}"; do
  echo "Running evaluation for ${model_path}"
  olmo-cookbook-eval evaluate "${model_path}" \
    --tasks mmlu:mc \
    --priority high \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --no-compute-gold-bpb \
    --model-args "trust_remote_code=true,max_length=4096" \
    --use-gantry \
    --gantry-args env-secret="OPENAI_API_KEY=openai_api_key" \
    --task-args 'chat_overrides={"generation_kwargs":{"stop_sequences":["Problem:","Answer:","Question:","</s>","<|eot_id|>"]}}' \
    --fim-tokens l2c \
    --vllm-use-v1-spec \
    --dashboard yapeic/proc \
    --workspace ai2/oe-data
done
