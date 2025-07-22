step=4769
models=(
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-12T-10B-1337-58f924c5"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-12T-10B-2032-904d0c69"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-12T-10B-3999-5bfc5eb1"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-12T-10B-pstar-v20-1337-9d1068ae"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-12T-10B-pstar-v20-2032-97053cd3"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-12T-10B-pstar-v20-3999-61a074d7"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-12T-10B-1337-6a9effc5"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-12T-10B-1337-8444268d"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-12T-10B-2032-c130d128"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-12T-10B-2032-d5f0ad8d"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-12T-10B-3999-1c01e4e6"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-12T-10B-3999-906bb3eb"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-equal-12T-10B-1337-709a3397"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-equal-12T-10B-2032-68f6b549"
    "/ai2-llm/checkpoints/lucas/olmo3_7b_with-reasoning-anneal-round2mix-v18-20-equal-12T-10B-3999-d4979c25"
)

to_add_pstar=(
    "6a9effc5"
    "d5f0ad8d"
    "1c01e4e6"
)

# Moving checkpoints to weka
for model in "${models[@]}"; do
    model_hex_suffix=$(echo $model | grep -oE '[0-9a-f]{8}')
    echo $model_hex_suffix
    if [[ " ${to_add_pstar[*]} " =~ " ${model_hex_suffix} " ]]; then
        output_model=$(echo $model | sed 's/12T-10B/12T-10B-pstar/')
    else
        output_model=$model
    fi

    uv run --python 3.12 python -m cookbook.remote "gs://${model}/step${step}" "weka://oe-training-default/${output_model}/step${step}" --allow-dirty
done


# Convert checkpoints
for model in "${models[@]}"; do
    model_hex_suffix=$(echo $model | grep -oE '[0-9a-f]{8}')
    if [[ " ${to_add_pstar[*]} " =~ " ${model_hex_suffix} " ]]; then
        to_convert_model=$(echo $model | sed 's/12T-10B/12T-10B-pstar/')
    else
        to_convert_model=$model
    fi

    uv run --python 3.12 \
        olmo-cookbook-eval convert \
        "/oe-training-default/${to_convert_model}/step${step}" \
        -t olmo-core-v2 \
        --use-beaker \
        --olmo-core-v2-commit-hash  013ef7b54aa2d583f9811ec6211a536da407a4b1 \
        --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
        --huggingface-transformers-commit-hash ca728b8879ce5127ea3e2f8d309c2c5febab5dc5
done

# Launch regular evals
dashboard="olmo3-midtraining-web"
for model in "${models[@]}"; do
    model_hex_suffix=$(echo $model | grep -oE '[0-9a-f]{8}')
    if [[ " ${to_add_pstar[*]} " =~ " ${model_hex_suffix} " ]]; then
        to_eval_model=$(echo $model | sed 's/12T-10B/12T-10B-pstar/')
    else
        to_eval_model=$model
    fi

    uv run olmo-cookbook-eval evaluate \
        "/oe-training-default/${to_eval_model}/step${step}-hf" \
        --tasks olmo3:dev:7b:main \
        --priority high \
        --cluster aus80g \
        --partition-size 8 \
        --num-gpus 1 \
        --model-backend vllm \
        --model-args trust_remote_code=true,max_length=4096 \
        --beaker-image oe-eval-beaker/oe_eval_qk_norm_auto \
        --dashboard ${dashboard} \
        --workspace ai2/oe-data
done
