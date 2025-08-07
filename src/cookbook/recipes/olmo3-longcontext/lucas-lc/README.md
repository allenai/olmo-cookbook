# Experiments

## Move

```bash
models=(
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1_bs16M-bac53cbd/step597"
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-qwen3like_olmo3mix_t8M-5ff97ac3/step4769"
    "ai2-llm/checkpoints/lucas/olmo3_7B-7T_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1_bs16M-028f5917/step597"
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1_t8M-ff85cc8c/step4769"
)

models=(
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-8k-64k_midtrain-with-reasoning_12T-eb8d9932/step2385"
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-8k-64k_olmo3mix_12T-51da76dd/step2385"
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning_12T-efbfbe1d/step2385"
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-qwen3like_olmo3mix_12T-02361a85/step2385"
)

# "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_12T-midtrain_round3_qwenlike_s2pdf_10B-e7d844e5/step2385"

models=(
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_12T-midtrain_round1_qwenlike_s2pdf_10B-682a95a4/step2385"
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_12T-midtrain_round1_qwenlike_s2pdf_20B-1b63158c/step4769"
)


models=(
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_12T-midtrain_round3_qwenlike_s2pdf_gzip2080_20B-2ad4d832/step4769"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_12T-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-172f9af6/step2385"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_12T-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preanneal-e16314e9/step2385"
)

models_do_not_use=(
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_12T-midtrain_olmo3mix_qwenlike_s2pdf_gzip2080_10B-73200adf/step2385"
)


for model in "${models[@]}"; do
    uv run --python 3.12 python -m cookbook.remote \
    gs://${model} \
    weka://oe-training-default/${model} \
    --allow-dirty
done
```


## Convert

```bash
for model in "${models[@]}"; do
    uv run --python 3.12 olmo-cookbook-eval convert \
        "/oe-training-default/${model}" \
        -t olmo-core-v2 \
        --use-beaker \
        --olmo-core-v2-commit-hash 71aa590af8d3979125cd7d96eb661d05e26d04a1 \
        --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
        --huggingface-transformers-commit-hash 4f2fbde7eaa7253b1ca977e294da6c9fcddfa345 \
        --dtype float32 --beaker-allow-dirty
done
```


## Eval (LC)

```bash
git clone git@github.com:allenai/HELMET.git code/ai2-helmet
cd code/ai2-helmet
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install beaker-gantry
source .venv/bin/activate

for model in "${models[@]}"; do
    TIMEOUT=0 PRIORITY=urgent NUM_GPUS=4 WORKSPACE=ai2/oe-data MODEL_NAME_OR_PATH=/weka/oe-training-default/${model}-hf CLUSTER=ai2/jupiter-cirrascale-2  ./gantry_eval.sh -r new
done
```


## Eval (SC)

```bash
dashboard="olmo3-long-context"
uv run olmo-cookbook-eval evaluate \
    "/oe-training-default/${model}-hf" \
    --tasks olmo3:dev:7b:main \
    --priority urgent \
    --cluster aus80g \
    --partition-size 4 \
    --num-gpus 1 \
    --model-backend vllm \
    --model-args trust_remote_code=true,max_length=4096 \
    --use-gantry \
    --dashboard ${dashboard} \
    --oe-eval-branch davidh/olmo3 \
    --beaker-image oe-eval-beaker/oe_eval_olmo3_auto \
    --workspace ai2/oe-data
```
