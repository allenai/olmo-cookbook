#!/bin/bash
set -e


# Usage: ./run_task_baseline_sampler.sh <task> <dashboard> [<priority>] [<cluster>]

TASK="$1"
DASHBOARD=$2
PRIORITY=${3:-low}
CLUSTER=${4:-aus80g}

echo "Running task: $TASK"

# List of models to loop over
THIRD_PARTY_MODELS=(
    "deepseek-ai/deepseek-llm-7b-base"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3.1-8B"
    # "Qwen/Qwen2.5-7B"
)

VLLM_MODELS=(
    "allenai--OLMo-2-0425-1B--stage2-ingredient3-step20000-tokens42B"
    "allenai--OLMo-2-0425-1B--stage2-ingredient3-step21000-tokens45B"
    "allenai--OLMo-2-0425-1B--stage2-ingredient3-step22000-tokens47B"
    "allenai--OLMo-2-0425-1B--stage2-ingredient3-step23000-tokens49B"
    "allenai--OLMo-2-0425-1B--stage2-ingredient3-step23852-tokens51B"
    "allenai--OLMo-2-1124-13B--stage2-ingredient4-step32000-tokens269B"
    "allenai--OLMo-2-1124-13B--stage2-ingredient4-step33000-tokens277B"
    "allenai--OLMo-2-1124-13B--stage2-ingredient4-step34000-tokens286B"
    "allenai--OLMo-2-1124-13B--stage2-ingredient4-step35000-tokens294B"
    "allenai--OLMo-2-1124-13B--stage2-ingredient4-step35773-tokens300B"
    "allenai--OLMo-2-1124-7B--stage2-ingredient3-step8000-tokens34B"
    "allenai--OLMo-2-1124-7B--stage2-ingredient3-step9000-tokens38B"
    "allenai--OLMo-2-1124-7B--stage2-ingredient3-step10000-tokens42B"
    "allenai--OLMo-2-1124-7B--stage2-ingredient3-step11000-tokens47B"
    "allenai--OLMo-2-1124-7B--stage2-ingredient3-step11931-tokens50B"
)

HF_MODELS=(
    "allenai/DataDecide-dolma1_7-1B"
    "allenai/DataDecide-dclm-baseline-50p-dolma1.7-50p-1B"
    "allenai/DataDecide-dclm-baseline-1B"
)

for MODEL in "${THIRD_PARTY_MODELS[@]}"; do
    olmo-cookbook-eval evaluate \
        "$MODEL" \
        --tasks "$TASK" \
        --priority "$PRIORITY" \
        --cluster "$CLUSTER" \
        --model-backend vllm \
        --dashboard "$DASHBOARD"
done

for MODEL in "${VLLM_MODELS[@]}"; do
    olmo-cookbook-eval evaluate \
        "/weka-mount/oe-eval-default/olmo-cookbook-baseline-sampler/$MODEL" \
        --tasks "$TASK" \
        --priority "$PRIORITY" \
        --cluster "$CLUSTER" \
        --model-backend vllm \
        --dashboard "$DASHBOARD"
done
for MODEL in "${HF_MODELS[@]}"; do
    olmo-cookbook-eval evaluate \
        "$MODEL" \
        --tasks "$TASK" \
        --priority "$PRIORITY" \
        --cluster "$CLUSTER" \
        --model-backend hf \
        --dashboard "$DASHBOARD"
done

