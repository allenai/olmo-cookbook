
##format for weka path: "/oe-training-default/ai2-llm/.../step12345-hf"

olmo-cookbook-eval evaluate \
  $1 \
  --tasks olmo3:dev:7b:main \
  --priority high \
  --cluster aus80g \
  --partition-size 8 \
  --num-gpus 1 \
  --model-backend vllm \
  --model-args trust_remote_code=true,max_length=4096 \
  --beaker-image oe-eval-beaker/oe_eval_qk_norm_auto \
  --dashboard olmo3-macroanneals \
  --workspace ai2/oe-data

olmo-cookbook-eval evaluate \
  $1 \
  --tasks olmo3:dev:midtrain:v0 \
  --priority high \
  --cluster aus80g \
  --num-gpus 1 \
  --partition-size 8 \
  --model-backend vllm \
  --no-compute-gold-bpb \
  --model-args chat_template=basic_answer,trust_remote_code=true,max_length=8192 \
  --use-gantry \
  --gantry-args env-secret="OPENAI_API_KEY=openai_api_key" \
  --task-args chat_overrides="{\"generation_kwargs\": {\"stop_sequences\": [\"Problem:\", \"Answer:\", \"Question:\", \"</s>\", \"<|eot_id|>\"]}}" \
  --oe-eval-branch davidh/head-qk-norm \
  --beaker-image oe-eval-beaker/oe_eval_qk_norm_auto \
  --dashboard olmo3-macroanneals \
  --workspace ai2/oe-data

# for model in olmo2-7b_5b-anneal_web-reasoning-47e0ff3c/step2385 olmo2-7b_10b-anneal_web-code-df914046/step4769 olmo2-7b_10b-anneal_web-code-reasoning-fbc68b5d/step4769 olmo2-7b_10b-anneal_web-math-c596e473/step4769 olmo2-7b_10b-anneal_web-math-reasoning-a5c1c043/step4769 olmo2-7b_10b-anneal_web-reddit-ac1d7ff4/step4769 olmo2-7b_10b-anneal_web-reddit-reasoning-ca5e1b62/step4769; do
#     olmo-cookbook-eval evaluate \
#         "/oe-training-default/ai2-llm/checkpoints/allysone/${model}-hf" \
#         --tasks mmlu:cot::reasoning --tasks mmlu:cot::none\
#         --priority urgent \
#         --cluster aus80g \
#         --num-gpus 1 \
#         --model-backend vllm \
#         --dashboard olmo3-midtraining \
#         --partition-size 1 \
#         --no-compute-gold-bpb

# done

# for model in olmo2-7b_5b-anneal_web-only-cc0d82b6/step2385 olmo2-7b_5b-anneal_web-reasoning-47e0ff3c/step2385 olmo2-7b_10b-anneal_web-code-df914046/step4769 olmo2-7b_10b-anneal_web-code-reasoning-fbc68b5d/step4769 olmo2-7b_10b-anneal_web-math-c596e473/step4769 olmo2-7b_10b-anneal_web-math-reasoning-a5c1c043/step4769 olmo2-7b_10b-anneal_web-reddit-ac1d7ff4/step4769 olmo2-7b_10b-anneal_web-reddit-reasoning-ca5e1b62/step4769; do
#     echo /path/to/${model}-hf
# done
