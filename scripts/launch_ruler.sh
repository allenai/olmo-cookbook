#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi


# Check if olmo-cookbook-eval command is available
if command -v olmo-cookbook-eval &> /dev/null; then
  eval_command="olmo-cookbook-eval"
elif command -v uv &> /dev/null && uv run olmo-cookbook-eval --help &> /dev/null; then
  eval_command="uv run olmo-cookbook-eval"
else
  echo "Error: olmo-cookbook-eval command not found. Please install it or ensure uv is available."
  exit 1
fi


model_path="$1"
base_command="${eval_command} evaluate \"${model_path}\" --priority urgent --cluster ai2/jupiter-cirrascale-2 --num-gpus 1 --model-backend vllm --dashboard peteish-LC-ruler --budget ai2/oe-base --model-args \"trust_remote_code=true,  chat_model=null, max_length=65536\"  --task-args \"use_chat_format=false\"  --vllm-use-v1-spec  --workspace ai2/long-contexts  --beaker-image amandab/lc-only-adjust-rope-global-layers"


echo "Launching task: ruler:4k"
eval "${base_command} --tasks ruler:4k -j 2"

echo "Launching task: ruler:8k"
eval "${base_command} --tasks ruler:8k -j 2"

echo "Launching task: ruler:16k"
eval "${base_command} --tasks ruler:16k -j 2"

echo "Launching task: ruler:32k"
eval "${base_command} --tasks ruler:32k -j 2"

echo "Launching task: ruler:64k"
eval "${base_command} --tasks ruler:64k -j 2"

wait
