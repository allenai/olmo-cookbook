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


# trying earlier signals for helmet
models=(
    "ai2-llm/checkpoints/lucas/olmo29_7b_lc_64k_1T-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preanneal-0a84103f/step2385"
    "ai2-llm/checkpoints/lucas/olmo29_7b_lc_64k_1T-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preanneal-highLR-60881d2e/step2385"
    "ai2-llm/checkpoints/lucas/olmo29_7b_lc_64k_2T-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preanneal-3f8d6d04/step2385"
    "ai2-llm/checkpoints/lucas/olmo29_7b_lc_64k_2T-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preanneal-highLR-f16fb41d/step2385"
    "ai2-llm/checkpoints/lucas/olmo29_7b_lc_64k_500B-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preanneal-highLR-8013d98e/step2385"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_1T-midtrain_olmo3mix_qwenlike_s2pdf_gzip2080_10B-preanneal-7d3eef66/step2385"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_500B-midtrain_olmo3mix_qwenlike_s2pdf_gzip2080_10B-preanneal-2531084a/step2385"
)

# does 2x batch size hurt?
models_borked=(
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2.5T-midtrain_olmo3mix_qwenlike_s2pdf_gzip2080_10B-preanneal-eb4860c6/step2385"
    "ai2-llm/checkpoints/lucas/olmo25-2xbzs_7b_lc_64k_2.5T-midtrain_olmo3mix_qwenlike_s2pdf_gzip2080_10B-preanneal-474bfe7b/step2385"
)
models=(
    "ai2-llm/checkpoints/lucas/olmo25-2xbzs_7b_lc_64k_2.5T-midtrain_olmo3mix_qwenlike_s2pdf_gzip2080_10B-preanneal-mixfix-eb6c19d0/step2385"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2.5T-midtrain_olmo3mix_qwenlike_s2pdf_gzip2080_10B-preanneal-mixfix-7949bf4b/step2385"
)

# the code one
models=(
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-SC_round5-LC_s2pdf_code-10B-ec413d34/step2385"
)

# round 5
models=(
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-SC33_round5-LC67_s2pdf_gzip2080-10B-a71e7b1d/step2385"
)


# 1B proxies
models=(
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-SC_round5-LC_s2pdf_code-1B-4M20W-ae66b812/step239"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-midtrain_round3_qwenlike_s2pdf_gzip2080_1B-4M20W-9054cc59/step239"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-SC33_round5-LC67_s2pdf_gzip2080-1B-4M20W-d4d68891/step239"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-SC30_r5s_nc-LC67_s2pdf_gzip2080_code30_pstar-1B-4M20W-ca1977e3/step239"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-SC33_round5slim-LC67_s2pdf_gzip2080_pstar-1B-4M20W-c253267a/step239"
)

# leftover to eval
models=(
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-SC30_r5s_nc-LC70_s2pdf_gzip2080_code30_pstar-10B-liteRep-9280996b/step2385"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T-SC33_round5slim-LC67_s2pdf_gzip2080_pstar-10B-fe8c2599/step2385"
)

# 4T + 100B + rewarm
models=(
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T_M100B_r5-midtrain_round3_qwenlike_s2pdf_gzip2080_50B_SC-again-R5-temp-pack-0204e0e9/step4769"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_2T_M100B_r5-midtrain_round3_qwenlike_s2pdf_gzip2080_100B-082001b9/step23842"
)

# these are all olmo-core models
models=(
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_4T_M100B_r5-midtrain_round3_qwenlike_s2pdf_gzip2080_yarn-fullonly_10B-09e2f9a1/step2385"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_4T_M100B_r5-midtrain_round3_qwenlike_s2pdf_gzip2080_inst-reas-synth_yarn-fullonly_10B-a952512f/step2385"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_4T_M100B_r5-midtrain_round3_qwenlike_s2pdf_gzip2080_inst-reas-synth-nocode_yarn-fullonly_10B-114cfbc0/step2385"
    "ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_4T_M100B_r5-midtrain_round3_qwenlike_s2pdf_gzip2080_just-synth_yarn-fullonly_10B-7b2e6b54/step2385"
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
        --olmo-core-v2-commit-hash 59465108d4214595083ab331233f86cd75125dce \
        --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
        --huggingface-transformers-commit-hash 4f2fbde7eaa7253b1ca977e294da6c9fcddfa345 \
        --dtype float32 \
        --beaker-allow-dirty \
        --skip-validation
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
    TIMEOUT=0 PRIORITY=urgent NUM_GPUS=8 WORKSPACE=ai2/oe-long-contexts MODEL_NAME_OR_PATH=/weka/oe-training-default/${model}-hf CLUSTER=ai2/jupiter-cirrascale-2  ./gantry_eval.sh -r new
done
```

Pull results

```bash
urls=(
    'https://beaker.allen.ai/orgs/ai2/workspaces/oe-data/work/01K2D3F101BN9WETC6WK7GZGSE?'
    'https://beaker.allen.ai/orgs/ai2/workspaces/oe-data/work/01K2CZMBAJ236C5AH5MEXWA0C7?'
    'https://beaker.allen.ai/orgs/ai2/workspaces/oe-data/work/01K2CZM7B7H7H6J3ZS6SBN9YZ8?'
    'https://beaker.allen.ai/orgs/ai2/workspaces/oe-data/work/01K2CZM5BHH918S8468ACA6H4W?'
    'https://beaker.allen.ai/orgs/ai2/workspaces/oe-data/work/01K2CZM38EW4XET2B1FDW8S9T5?'
    'https://beaker.allen.ai/orgs/ai2/workspaces/oe-data/work/01K2CZM19H0SKYBZNR02Y1V5ZX?'
    'https://beaker.allen.ai/orgs/ai2/workspaces/oe-data/work/01K2CZKZ88ZY33YGMKY3F0B0VJ?'
)

for url in "${urls[@]}"; do
    ./fetch_results.py -u ${url} >> ~/Downloads/results.csv
done
```

For ruler:

```bash
for model in "${models[@]}"; do
    ./scripts/launch_ruler.sh /oe-training-default/${model}-hf
done
```



## Eval (SC)

```bash
for model in "${models[@]}"; do
    ./scripts/launch_sc4lc.sh /oe-training-default/${model}-hf
done
```

## Eval'ing core models

Short context

```bash
for model in "${models[@]}"; do
    ./scripts/launch_sc4lc.sh /oe-training-default/${model}
done
```

RULER

```bash
for model in "${models[@]}"; do
    ./scripts/launch_ruler.sh /oe-training-default/${model}
done
```

Fetch results

```bash
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p temp
for model in "${models[@]}"; do
    uv run --python 3.12 ./scripts/grab_ruler.py $(echo ${model} | sed -E 's#.*/([^/]+)/([^/]+)$#\1_\2#') >> temp/ruler_results_${timestamp}.csv
done
```
