#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi

LITE_TASKS=(
  "gen::xlarge"
  "mbpp:3shot::olmo3:n32:v2"
  "minerva"
  "olmo3:dev:7b:mcqa:stem"
  "olmo3:dev:7b:mcqa:non_stem"
)

# Check if olmo-cookbook-eval command is available
if command -v olmo-cookbook-eval &> /dev/null; then
  eval_command="olmo-cookbook-eval"
elif command -v uv &> /dev/null && uv run olmo-cookbook-eval --help &> /dev/null; then
  eval_command="uv run olmo-cookbook-eval"
else
  echo "Error: olmo-cookbook-eval command not found. Please install it or ensure uv is available."
  exit 1
fi

tasks=""
for task in "${LITE_TASKS[@]}"; do
  tasks+=" --tasks ${task}"
done

${eval_command} evaluate \
  "$1" \
  --priority urgent \
  --cluster aus80g \
  --partition-size 4 \
  --num-gpus 1 \
  --model-backend vllm \
  --dashboard peteish-LC-ruler \
  --budget ai2/oe-base \
  --model-args "trust_remote_code=true,  chat_model=null, max_length=65536" \
  --task-args "use_chat_format=false" \
  --vllm-use-v1-spec \
  --workspace ai2/oe-data \
  --beaker-image amandab/lc-only-adjust-rope-global-layers \
  ${tasks}
