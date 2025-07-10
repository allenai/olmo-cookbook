# Experiments

## Move

```bash
model="olmo3_7b_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1-b148c0f9/step4769"
model="olmo3_7b_lc_64k_s2pdf-qwen3like_olmo3mix-62a97530/step4769"
model="olmo3_7b_lc_64k_s2pdf-8k-64k_midtrain-with-reasoning-w1-ccb7b996/step4769"
model="olmo3_7b_lc_64k_s2pdf-8k-64k_olmo3mix-f60303aa/step4769"

uv run python -m cookbook.remote \
    gs://ai2-llm/checkpoints/$(whoami)/${model} \
    weka://oe-training-default/ai2-llm/checkpoints/$(whoami)/${model}
```


## Convert



```bash
model="olmo3_7b_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1-b148c0f9/step4769"
model="olmo3_7b_lc_64k_s2pdf-qwen3like_olmo3mix-62a97530/step4769"
model="olmo3_7b_lc_64k_s2pdf-8k-64k_midtrain-with-reasoning-w1-ccb7b996/step4769"
model="olmo3_7b_lc_64k_s2pdf-8k-64k_olmo3mix-f60303aa/step4769"

olmo-cookbook-eval convert \
    "/oe-training-default/ai2-llm/checkpoints/$(whoami)/${model}" \
    -t olmo-core-v2 \
    --use-beaker \
    --olmo-core-v2-commit-hash  326b7b01cc77750343510919801316d5a5622d87 \
    --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
    --huggingface-transformers-commit-hash 5db7e35d42636e86ee37a43f56a1587daadb7c1b
```
