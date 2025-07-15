# Experiments

## Move

```bash
models=(
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1_bs16M-bac53cbd/step597"
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-qwen3like_olmo3mix_t8M-5ff97ac3/step4769"
    "ai2-llm/checkpoints/lucas/olmo3_7B-7T_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1_bs16M-028f5917/step597"
    "ai2-llm/checkpoints/lucas/olmo3_7b_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1_t8M-ff85cc8c/step4769"
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

olmo-cookbook-eval convert \
    "/oe-training-default/ai2-llm/checkpoints/$(whoami)/${model}" \
    -t olmo-core-v2 \
    --use-beaker \
    --olmo-core-v2-commit-hash  326b7b01cc77750343510919801316d5a5622d87 \
    --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
    --huggingface-transformers-commit-hash 5db7e35d42636e86ee37a43f56a1587daadb7c1b
```
