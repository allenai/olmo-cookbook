models=(
    "/ai2-llm/checkpoints/lucas/olmo3-microanneal-50web-50wiki-8a9b662c/step4769"   # rewritten, 10x rep
    "/ai2-llm/checkpoints/lucas/olmo3-microanneal-50web-50wiki-2618a292/step4769"   # as-is, 2x rep
)

# Moving checkpoints to weka
for model in "${models[@]}"; do
    uv run --python 3.12 python -m cookbook.remote gs://$model weka://oe-training-default/$model --allow-dirty
done


# Convert checkpoints
for model in "${models[@]}"; do
    uv run --python 3.12 \
        olmo-cookbook-eval convert \
        "/oe-training-default/$model" \
        -t olmo-core-v2 \
        --use-beaker \
        --olmo-core-v2-commit-hash  013ef7b54aa2d583f9811ec6211a536da407a4b1 \
        --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
        --huggingface-transformers-commit-hash ca728b8879ce5127ea3e2f8d309c2c5febab5dc5
done



# Launch regular evals
dashboard="olmo3-midtraining-web"
for model in "${models[@]}"; do
    uv run olmo-cookbook-eval evaluate \
        "/oe-training-default/${model}-hf" \
        --tasks  olmo3:dev:7b:gen \
        --priority high \
        --cluster aus80g \
        --partition-size 4 \
        --num-gpus 1 \
        --model-backend vllm \
        --model-args trust_remote_code=true,max_length=4096 \
        --beaker-image oe-eval-beaker/oe_eval_qk_norm_auto \
        --dashboard ${dashboard} \
        --workspace ai2/oe-data
done
