#!/bin/bash

# List of model names (just the model IDs from the URLs)
models=(
#"allenai/DataDecide-dolma1_7-1B"
#"allenai/DataDecide-dclm-baseline-25p-dolma1.7-75p-1B"
#"allenai/DataDecide-falcon-and-cc-qc-10p-1B"
#"allenai/DataDecide-falcon-1B"
#"allenai/DataDecide-dclm-baseline-qc-7p-fw3-1B"
#"allenai/DataDecide-dclm-baseline-qc-7p-fw2-1B"
#"allenai/DataDecide-dclm-baseline-1B"
#"allenai/DataDecide-falcon-and-cc-1B"
#"allenai/DataDecide-dclm-baseline-50p-dolma1.7-50p-1B"
#"allenai/DataDecide-falcon-and-cc-qc-orig-10p-1B"
#"allenai/DataDecide-falcon-and-cc-qc-20p-1B"
#"allenai/DataDecide-fineweb-edu-1B"
#"allenai/DataDecide-dolma1_7-no-math-code-1B"
#"allenai/DataDecide-fineweb-pro-1B"
#"allenai/DataDecide-falcon-and-cc-qc-tulu-10p-1B"
#"allenai/DataDecide-dolma1_6plus-1B"
"allenai/DataDecide-dclm-baseline-75p-dolma1.7-25p-1B"
"allenai/DataDecide-dolma1_7-no-code-1B"
"allenai/DataDecide-dolma1_7-no-flan-1B"
"allenai/DataDecide-dolma1_7-no-reddit-1B"
"allenai/DataDecide-c4-1B"
"allenai/DataDecide-dclm-baseline-qc-20p-1B"
"allenai/DataDecide-dclm-baseline-qc-fw-3p-1B"
#"allenai/DataDecide-dclm-baseline-qc-10p-1B"
#"allenai/DataDecide-dclm-baseline-qc-fw-10p-1B"
)

# Loop over all models
: 'for model in "${models[@]}"; do
  echo "Running evaluation for $model"
  olmo-cookbook-eval evaluate "$model" \
      --tasks arc_challenge:rc::olmes:full \
      --tasks arc_easy:rc::olmes:full \
      --tasks hellaswag:rc::olmes \
      --tasks winogrande:rc::olmes:full \
      --tasks csqa:rc::olmes:full \
      --tasks piqa:rc::olmes:full \
      --tasks socialiqa:rc::olmes:full \
      --tasks mmlu:rc::olmes \
      --tasks gsm8k::olmes \
      --tasks minerva_math::olmes \
      --tasks codex_humaneval:temp0.1:bpb \
      --tasks mbpp:bpb::none \
      --tasks basic_skills:rc::olmes \
      --tasks mt_mbpp \
      --priority high \
      --cluster aus80g \
      --num-gpus 1 \
      --dashboard regmixer \
      --compute-gold-bpb \
      --model-backend vllm \
      --workspace ai2/dolma2 \
      --model-args dtype=bfloat16
done'




: 'olmo-cookbook-eval evaluate "/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0000/step22100-hf" \
    --tasks arc_easy:rc::olmes:full \
    --tasks arc_challenge:rc::olmes:full \
    --tasks hellaswag:rc::olmes \
    --tasks winogrande:rc::olmes:full \
    --tasks csqa:rc::olmes:full \
    --tasks piqa:rc::olmes:full \
    --tasks socialiqa:rc::olmes:full \
    --tasks mmlu:rc \
    --tasks basic_skills:rc::olmes \
    --tasks minerva \
    --tasks codex_humaneval:3shot:bpb::none \
    --tasks mbpp:3shot:bpb::none \
    --tasks mt_mbpp \
    --tasks medmcqa:rc:bpb::none \
    --tasks lambada \
    --tasks sciq:bpb::olmo1 \
    --tasks squad:rc:bpb::gen2mc \
    --tasks naturalqs:rc:bpb::gen2mc  \
    --tasks jeopardy:rc:bpb::gen2mc \
    --tasks drop:rc:bpb::gen2mc \
    --tasks coqa:rc:bpb::gen2mc \
    --tasks ultrachat_masked_ppl \
    --tasks wildchat_masked_ppl \
    --compute-gold-bpb \
    --priority high \
    --cluster 80g \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args dtype=bfloat16 \
    --partition-size 8 \
    --dashboard regmixer  \
    --workspace ai2/dolma2 '







EXCLUDED=("0086" "0187")

for i in $(seq -f "%04g" 3 511); do
    # Skip excluded steps
    if [[ " ${EXCLUDED[@]} " =~ " ${i} " ]]; then
        echo "Skipping excluded step $i"
        continue
    fi

    CHECKPOINT="/oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-${i}/step22100-hf"

    echo "Evaluating checkpoint: $CHECKPOINT"

    olmo-cookbook-eval evaluate "$CHECKPOINT" \
        --tasks arc_easy:rc::olmes:full \
        --tasks arc_challenge:rc::olmes:full \
        --tasks hellaswag:rc::olmes \
        --tasks winogrande:rc::olmes:full \
        --tasks csqa:rc::olmes:full \
        --tasks piqa:rc::olmes:full \
        --tasks socialiqa:rc::olmes:full \
        --tasks mmlu:rc \
        --tasks basic_skills:rc::olmes \
        --tasks minerva \
        --tasks codex_humaneval:3shot:bpb::none \
        --tasks mbpp:3shot:bpb::none \
        --tasks mt_mbpp \
        --tasks medmcqa:rc:bpb::none \
        --tasks lambada \
        --tasks sciq:bpb::olmo1 \
        --tasks squad:rc:bpb::gen2mc \
        --tasks naturalqs:rc:bpb::gen2mc  \
        --tasks jeopardy:rc:bpb::gen2mc \
        --tasks drop:rc:bpb::gen2mc \
        --tasks coqa:rc:bpb::gen2mc \
        --compute-gold-bpb \
        --priority urgent \
        --cluster 80g \
        --num-gpus 1 \
        --model-backend vllm \
        --model-args dtype=bfloat16 \
        --partition-size 8 \
        --dashboard regmixer \
        --workspace ai2/dolma2
done
