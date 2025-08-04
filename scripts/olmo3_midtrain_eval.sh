
# Check if first argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path> <dashboard>"
    echo "Example: $0 /oe-training-default/ai2-llm/checkpoints/mattj/microanneal-dolminos_math_baseline-1B-ffabe337/step477-hf"
    exit 1
fi

# Store the first argument in a variable
CHECKPOINT_PATH="$1"
DASHBOARD="$2"

olmo-cookbook-eval evaluate \
  "$CHECKPOINT_PATH" \
  --tasks olmo3:dev:midtrain:v1 \
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
  --fim-tokens l2c \
  --oe-eval-branch davidh/olmo3 \
  --beaker-image oe-eval-beaker/oe_eval_olmo3_auto \
  --vllm-use-v1-spec \
  --dashboard $DASHBOARD \
  --workspace ai2/olmo-3-microanneals

olmo-cookbook-eval evaluate \
  "$CHECKPOINT_PATH" \
  --tasks olmo3:dev:7b:main:v2 \
  --priority high \
  --cluster aus80g \
  --partition-size 8 \
  --num-gpus 1 \
  --model-backend vllm \
  --model-args trust_remote_code=true,max_length=4096 \
  --beaker-image oe-eval-beaker/oe_eval_olmo3_auto\
  --fim-tokens l2c \
  --vllm-use-v1-spec \
  --vllm-memory-utilization 0.7 \
  --dashboard $DASHBOARD \
  --workspace ai2/olmo-3-microanneals

