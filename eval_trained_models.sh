#!/usr/bin/env bash
set -euo pipefail

paths_and_revs=(
    # "allenai/OLMo-2-1124-7B-Instruct,"
    # "/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/sft_olmo7b_v1__8__1760319619/epoch_0_hf,"
    # "/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/sft_olmo7b_v1__8__1760319619/epoch_1_hf,"
    # "/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/sft_olmo7b_v1__8__1760319619/epoch_2_hf,"
    # "allenai/OLMo-2-1124-7B,"
    # "/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/sft_olmo7b-base_v3__8__1760388742/sft_olmo7b-base_v3_epoch_0_hf,"
    # "yapeichang/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop,step100"
    # "yapeichang/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template,step200"
    # "/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template__1__1761404946_checkpoints/step_200,"
    # "Qwen/Qwen2.5-7B-Instruct,"
    "Qwen/Qwen3-4B"
    "yapeichang/grpo_qwen3-4b-inst_v1_binary,step_300"
)

# TODO: right now if use hf model with revision, tokenizer is not loaded correctly, currently i'm using the local weka model

# all_tasks=(
#   # "mmlu:cot::hamish_zs_reasoning"
#   # "gpqa:0shot_cot::hamish_zs_reasoning"
#   # "minerva_math::hamish_zs_reasoning"
#   # "gsm8k::hamish_zs_reasoning"
#   # "aime::hamish_zs_reasoning"
#   # "bbh:cot::hamish_zs_reasoning"
#   # "minerva_math_500::hamish_zs_reasoning"
#   # "agi_eval_english:0shot_cot::hamish_zs_reasoning"
#   # "ifeval::hamish_zs_reasoning"
#   # "popqa::hamish_zs_reasoning"
#   # "alpaca_eval_v3::hamish_zs_reasoning"
#   # "mbppplus:0-shot-chat::tulu-thinker"
#   # "codex_humanevalplus:0-shot-chat-v1::tulu-thinker"
#   # "gsm8k::zs_cot_latex"
#   "olmo3:adapt"
# )

# all_tasks=(
#   # "mmlu:cot::olmo3:adapt"
#   # "popqa::olmo3:adapt"
#   # "simpleqa::olmo3:adapt"
#   # "bbh:cot::olmo3:adapt"
#   # "gpqa::olmo3:adapt"
#   # "zebralogic::olmo3:adapt"
#   # "agi_eval_english::olmo3:adapt"
#   # "minerva_math::olmo3:adapt"
#   # "gsm8k::olmo3:adapt"
#   # "omega::olmo3:adapt"
#   # "aime:2024::olmo3:adapt"
#   # "aime:2025::olmo3:adapt"
#   # "codex_humanevalplus::olmo3:adapt"
#   # "mbppplus::olmo3:adapt"
#   # "livecodebench_codegeneration::olmo3:adapt"
#   # "alpaca_eval_v3::olmo3:adapt"
#   # "ifeval::olmo3:adapt"
#   # "multipl_e:6lang::olmo3"
#   # "multipl-e-humaneval:n32:v2"
# )

all_tasks=(
    # Knowledge
    "mmlu:cot::hamish_zs_reasoning_deepseek"
    "popqa::hamish_zs_reasoning_deepseek"
    "simpleqa::tulu-thinker_deepseek"
    
    # Reasoning
    "bbh:cot::hamish_zs_reasoning_deepseek_v2" # OLD: "bbh:cot::hamish_zs_reasoning_deepseek"
    "gpqa:0shot_cot::qwen3-instruct"
    "zebralogic::hamish_zs_reasoning_deepseek"
    "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek"

    # Math
    # [faster] minerva_math_500::hamish_zs_reasoning
    "minerva_math::hamish_zs_reasoning_deepseek"
    "gsm8k::zs_cot_latex_deepseek"
    "omega_500:0-shot-chat_deepseek" # OLD: "omega:0-shot-chat"
    "aime:zs_cot_r1::pass_at_32_2024_deepseek"
    "aime:zs_cot_r1::pass_at_32_2025_deepseek"  # OLD: "aime::hamish_zs_reasoning"
    
    # Coding
    "codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek"
    "mbppplus:0-shot-chat::tulu-thinker_deepseek"
    "livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"
    
    # Chat / IF / Vibes
    "alpaca_eval_v3::hamish_zs_reasoning_deepseek"
    "ifeval::hamish_zs_reasoning_deepseek"
)

for entry in "${paths_and_revs[@]}"; do
  for task in "${all_tasks[@]}"; do
    IFS=',' read -r model_path revision <<< "${entry}"

    echo "Running evaluation for ${model_path} (revision: ${revision:-'default'})"

    olmo-cookbook-eval evaluate "${model_path}" \
      ${revision:+--revision "$revision"} \
      --tasks "${task}" \
      --priority high \
      --cluster aus80g \
      --num-gpus 1 \
      --partition-size 8 \
      --model-backend vllm \
      --no-compute-gold-bpb \
      --model-args "process_output=r1_style,chat_template=tulu_thinker_r1_style,trust_remote_code=true,max_length=8192,dtype=bfloat16" \
      --task-args 'chat_overrides={"generation_kwargs":{"stop_sequences":["Problem:","Answer:","Question:","</s>","<|eot_id|>"]}}' \
      --fim-tokens l2c \
      --vllm-use-v1-spec \
      --beaker-image oe-eval-beaker/oe_eval_auto \
      --huggingface-secret yapeic_HF_TOKEN \
      --dashboard yapeic/train_new \
      --workspace ai2/oe-data
  done
done

# max_length=8192,
      # --oe-eval-branch yapeic/formatss \

# --tasks mmlu:cot::hamish_zs_reasoning \
    # --tasks gpqa:0shot_cot::hamish_zs_reasoning \
    # --tasks minerva_math::hamish_zs_reasoning \
    # --tasks bbh:cot::hamish_zs_reasoning \
    # --tasks gsm8k::hamish_zs_reasoning \
    # --tasks minerva_math_500::hamish_zs_reasoning \
    # --tasks aime::hamish_zs_reasoning \
    # --tasks agi_eval_english:0shot_cot::hamish_zs_reasoning \

# olmo-cookbook-eval results --tasks mmlu:cot::hamish_zs_reasoning --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks gpqa:0shot_cot::hamish_zs_reasoning --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks minerva_math::hamish_zs_reasoning --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks bbh:cot::hamish_zs_reasoning --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks gsm8k::hamish_zs_reasoning --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks minerva_math_500::hamish_zs_reasoning --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks aime::hamish_zs_reasoning --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks agi_eval_english:0shot_cot::hamish_zs_reasoning --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks olmo3:adapt --format csv --dashboard yapeic/train_new
# olmo-cookbook-eval results --tasks multipl-e-humaneval --format csv --dashboard yapeic/train_new