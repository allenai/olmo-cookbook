#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi


# Check if olmo-cookbook-eval command is available
if command -v olmo-cookbook-eval &> /dev/null; then
  eval_command="olmo-cookbook-eval"
elif command -v uv &> /dev/null && uv run olmo-cookbook-eval --help &> /dev/null; then
  eval_command="uv run --python=3.12 --extra=all  olmo-cookbook-eval"
else
  echo "Error: olmo-cookbook-eval command not found. Please install it or ensure uv is available."
  exit 1
fi

# if model path ends with -hf, then the backend is vllm; otherwise, it's olmo_core
if [[ "$1" == *"-hf" ]]; then
  backend="--model-backend vllm --vllm-use-v1-spec"
  beaker_image="--beaker-image amandab/lc-only-adjust-rope-global-layers"
  oe_eval_branch=""
else
  backend="--model-backend olmo_core"
  beaker_image="--beaker-image tylerr/oe_eval_olmocore_082725"
  oe_eval_branch="--oe-eval-commit 3d53a693a9236cbdb1bac0543b599e0bd7f3c2d7 --use-gantry"
fi

if [[ "$1" == "gs"* ]]; then
  # evaluate on augusta cluster
  cluster="ai2/augusta-google-1"
else
  cluster="ai2/jupiter-cirrascale-2"
fi

model_path="$1"
base_command="${eval_command} evaluate \"${model_path}\" --priority urgent --cluster ${cluster} --num-gpus 1 ${backend} --dashboard peteish-LC-ruler --budget ai2/oe-base --model-args \"trust_remote_code=true,  chat_model=null, max_length=65536\"  --task-args \"use_chat_format=false\" --workspace ai2/long-contexts ${beaker_image} ${oe_eval_branch}"


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
