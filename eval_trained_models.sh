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
    "/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template__1__1761404946_checkpoints/step_200,"
    # "Qwen/Qwen2.5-7B-Instruct,"
)

# TODO: right now if use hf model with revision, tokenizer is not loaded correctly, currently i'm using the local weka model

all_tasks=(
  # "mmlu:cot::hamish_zs_reasoning"
  # "gpqa:0shot_cot::hamish_zs_reasoning"
  # "minerva_math::hamish_zs_reasoning"
  # "gsm8k::hamish_zs_reasoning"
  # "aime::hamish_zs_reasoning"
  # "bbh:cot::hamish_zs_reasoning"
  # "minerva_math_500::hamish_zs_reasoning"
  # "agi_eval_english:0shot_cot::hamish_zs_reasoning"
  # "ifeval::hamish_zs_reasoning"
  # "popqa::hamish_zs_reasoning"
  # "alpaca_eval_v3::hamish_zs_reasoning"
  # "mbppplus:0-shot-chat::tulu-thinker"
  # "codex_humanevalplus:0-shot-chat-v1::tulu-thinker"
  # "gsm8k::zs_cot_latex"
  "olmo3:adapt"
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
      --model-args "process_output=r1_style,chat_template=tulu_thinker_r1_style,trust_remote_code=true,max_length=4096,dtype=bfloat16" \
      --task-args 'chat_overrides={"generation_kwargs":{"stop_sequences":["Problem:","Answer:","Question:","</s>","<|eot_id|>"]}}' \
      --fim-tokens l2c \
      --vllm-use-v1-spec \
      --beaker-image oe-eval-beaker/oe_eval_auto \
      --huggingface-secret yapeic_HF_TOKEN \
      --dashboard yapeic/train_new \
      --workspace ai2/oe-data
  done
done

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