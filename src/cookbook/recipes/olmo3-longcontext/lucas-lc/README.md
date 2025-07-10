# Experiments

## Convert

```bash
user=$(whoami)
model="olmo3_7b_lc_64k_s2pdf-qwen3like_midtrain-with-reasoning-w1-b148c0f9/step4769"

olmo-cookbook-eval convert \
    "/oe-training-default/ai2-llm/checkpoints/${user}/${model}" \
    -t olmo-core-v2 \
    --use-beaker \
    --olmo-core-v2-commit-hash  326b7b01cc77750343510919801316d5a5622d87 \
    --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
    --huggingface-transformers-commit-hash 5db7e35d42636e86ee37a43f56a1587daadb7c1b
```
