#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi

MAX_LENGTH=${MAX_LENGTH:-65536}


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


# if model path ends with -hf, then the backend is vllm; otherwise, it's olmo_core
if [[ "$1" == *"-hf" ]]; then
  backend="--model-backend vllm --vllm-use-v1-spec"
  beaker_image="--beaker-image amandab/lc-only-adjust-rope-global-layers"
  oe_eval_branch=""
else
  backend="--model-backend olmo_core"
  beaker_image="--beaker-image tylerr/oe_eval_olmocore_082725"
  oe_eval_branch="--oe-eval-branch tyler/olmocore-native-eval --use-gantry"
fi


${eval_command} evaluate \
  "$1" \
  --priority urgent \
  --cluster aus80g \
  --partition-size 4 \
  --num-gpus 1 \
  ${backend} \
  --dashboard peteish-LC-ruler \
  --budget ai2/oe-base \
  --model-args "trust_remote_code=true,  chat_model=null, max_length=${MAX_LENGTH}" \
  --task-args "use_chat_format=false" \
  --workspace ai2/long-contexts \
  ${beaker_image} \
  ${oe_eval_branch} \
  ${tasks}
