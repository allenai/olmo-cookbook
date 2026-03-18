MODELS=(
    "s3://ai2-llm/checkpoints/lucas/olmo3_1b_5xc_50web_alldressed_v2_50spring2code_stack_edu_redux_all/step61007-hf"
    "/oe-training-default/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995-hf"
    "/oe-training-default/kevinf/checkpoints/train-olmo3-1b-dolma50-stackedu-python50-10B-lr5e-5-ctd/step2385-hf"
)
DASHBOARD="kevinf-flex"

for MODEL_PATH in "${MODELS[@]}"; do
    echo "=== Launching eval for: ${MODEL_PATH} ==="
    uv run olmo-cookbook-eval evaluate \
        "${MODEL_PATH}" \
        --priority urgent \
        --cluster ai2/saturn \
        --num-gpus 1 \
        --model-backend vllm \
        --dashboard "${DASHBOARD}" \
        --budget ai2/oe-base \
        --vllm-use-v1-spec \
        --model-args 'trust_remote_code=true, chat_model=null, max_length=8192' \
        --task-args 'use_chat_format=false' \
        --partition-size 4 \
        --tasks code-no-bcb
done
