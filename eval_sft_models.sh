#!/usr/bin/env bash
set -euo pipefail

paths_and_revs=(
    "yapeichang/sft_olmo3-1025-7b_mix_0-100,"
    "yapeichang/sft_olmo3-1025-7b_mix_5-95,"
    "yapeichang/sft_olmo3-1025-7b_mix_10-90,"
    # "yapeichang/sft_qwen3-8b-base_mix_0-100,"
    # "yapeichang/sft_qwen3-8b-base_mix_5-95,"
    # "yapeichang/sft_qwen3-8b-inst_v1_simple_with_mix"
)

all_tasks=(
    # "gpqa:0shot_cot::qwen3-instruct"
    # "zebralogic::hamish_zs_reasoning_deepseek"
    # # "minerva_math::hamish_zs_reasoning_deepseek"
    # "gsm8k::zs_cot_latex_deepseek"
    # "codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek"
    # "mbppplus:0-shot-chat::tulu-thinker_deepseek"
    # # "livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"
    # "alpaca_eval_v3::hamish_zs_reasoning_deepseek"
    # # "aime:zs_cot_r1::pass_at_32_2024_deepseek"
    # "aime:zs_cot_r1::pass_at_32_2025_deepseek"
    # "omega_500:0-shot-chat_deepseek"
    # "mmlu_pro:0shot_cot::tulu3"
    # # "mmlu:cot::hamish_zs_reasoning_deepseek"
    "ifbench::tulu"
)

max_tokens=32768

for entry in "${paths_and_revs[@]}"; do
  for task in "${all_tasks[@]}"; do
    IFS=',' read -r model_path revision <<< "${entry}"

    echo "Running evaluation for ${model_path} (revision: ${revision:-'default'})"

    olmo-cookbook-eval evaluate "${model_path}" \
      ${revision:+--revision "$revision"} \
      --tasks "${task}" \
      --priority normal \
      --cluster aus80g \
      --num-gpus 1 \
      --partition-size 8 \
      --model-backend vllm \
      --no-compute-gold-bpb \
      --model-args "trust_remote_code=true,max_length=${max_tokens},dtype=bfloat16" \
      --task-args 'chat_overrides={"generation_kwargs":{"stop_sequences":["Problem:","Question:","<|im_end|>"]}}' \
      --gantry-args env-secret="OPENAI_API_KEY=yapeic_OPENAI_API_KEY" \
      --fim-tokens l2c \
      --vllm-use-v1-spec \
      --beaker-image oe-eval-beaker/oe_eval_auto \
      --huggingface-secret yapeic_HF_TOKEN \
      --dashboard yapeic/train_r2 \
      --workspace ai2/oe-data
  done
done

# --task-args 'chat_overrides={"generation_kwargs":{"stop_sequences":["Problem:","Answer:","Question:","<|im_start|>","<|im_end|>"]}}' \