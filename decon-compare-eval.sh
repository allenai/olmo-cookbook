# olmo-cookbook-eval evaluate \
#   "/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round5-100B-olmo25_7b-anneal-6T-decon-sparkle-motion-8730626c/step47684-hf" \
#   --tasks olmo3:dev:midtrain:v1 \
#   --priority normal \
#   --cluster ai2/jupiter \
#   --num-gpus 1 \
#   --partition-size 8 \
#   --model-backend vllm \
#   --no-compute-gold-bpb \
#   --model-args chat_template=basic_answer,trust_remote_code=true,max_length=8192 \
#   --use-gantry \
#   --gantry-args env-secret="OPENAI_API_KEY=openai_api_key" \
#   --task-args chat_overrides="{\"generation_kwargs\": {\"stop_sequences\": [\"Problem:\", \"Answer:\", \"Question:\", \"</s>\", \"<|eot_id|>\"]}}" \
#   --fim-tokens l2c \
#   --oe-eval-branch davidh/olmo2-retrofit \
#   --beaker-image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
#   --vllm-use-v1-spec \
#   --dashboard ianm-decon-compare \
#   --workspace ai2/olmo-3-microanneals


olmo-cookbook-eval evaluate \
  "/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round5-100B-olmo25_7b-anneal-6T-decon-sparkle-motion-8730626c/step47684-hf" \
  --tasks olmo3:dev:7b:main:v2 \
  --priority normal \
  --cluster aus80g \
  --partition-size 8 \
  --num-gpus 1 \
  --model-backend vllm \
  --model-args trust_remote_code=true,max_length=4096 \
  --beaker-image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
  --fim-tokens l2c \
  --vllm-use-v1-spec \
  --vllm-memory-utilization 0.7 \
  --dashboard ianm-decon-compare \
  --workspace ai2/olmo-3-microanneals



olmo-cookbook-eval evaluate \
  "/oe-training-default/ai2-llm/checkpoints/ianm/anneal-round5-100B-olmo25_7b-anneal-6T-non-decon-fixed-ed30fcae/step47684-hf" \
  --tasks olmo3:dev:7b:main:v2 \
  --priority normal \
  --cluster ai2/jupiter \
  --num-gpus 1 \
  --partition-size 8 \
  --model-backend vllm \
  --no-compute-gold-bpb \
  --model-args chat_template=basic_answer,trust_remote_code=true,max_length=8192 \
  --use-gantry \
  --gantry-args env-secret="OPENAI_API_KEY=openai_api_key" \
  --task-args chat_overrides="{\"generation_kwargs\": {\"stop_sequences\": [\"Problem:\", \"Answer:\", \"Question:\", \"</s>\", \"<|eot_id|>\"]}}" \
  --fim-tokens l2c \
  --oe-eval-branch davidh/olmo2-retrofit \
  --beaker-image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
  --vllm-use-v1-spec \
  --dashboard ianm-decon-compare \
  --workspace ai2/olmo-3-microanneals


olmo-cookbook-eval evaluate \
  "/oe-training-default/ai2-llm/checkpoints/ianm/anneal-round5-100B-olmo25_7b-anneal-6T-non-decon-fixed-ed30fcae/step47684-hf" \
  --tasks olmo3:dev:7b:main:v2 \
  --priority normal \
  --cluster aus80g \
  --partition-size 8 \
  --num-gpus 1 \
  --model-backend vllm \
  --model-args trust_remote_code=true,max_length=4096 \
  --beaker-image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
  --fim-tokens l2c \
  --vllm-use-v1-spec \
  --vllm-memory-utilization 0.7 \
  --dashboard ianm-decon-compare \
  --workspace ai2/olmo-3-microanneals


#   olmo-cookbook-eval convert "/oe-training-default/ai2-llm/checkpoints/ianm/anneal-round5-100B-olmo25_7b-anneal-6T-non-decon-fixed-ed30fcae/step47684" \
#  -t olmo-core-v2 \
#  --use-beaker \
#  --olmo-core-v2-commit-hash 8d29c9edcc0739121e11e84f072776c4ff76b2d3 \
#  --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
#  --huggingface-transformers-commit-hash 4f2fbde7eaa7253b1ca977e294da6c9fcddfa345

