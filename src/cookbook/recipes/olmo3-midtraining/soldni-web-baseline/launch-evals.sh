#!/usr/bin/env bash

models=(
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-all-dressed-v18-ce7c0c2a/step476'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-all-dressed-v20-1e29004d/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-hq-web-baseline-0e513fe6/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-actual-5d35521f/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-distill-a2d7cd25/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-extract-knowledge-76f1ff02/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-knowledge_list-de0e6931/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-qa-d2535212/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-wrap_medium-ab332b66/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-olmo3-mix-2b9cd813/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-microanneal-round1-olmo3_7b-with-reasoning-anneal-03fb7a8d/step4769'
    'ai2-llm/checkpoints/lucas/olmo3-mix-1b-5xC-b5ad93d5/step4769'
    'ai2-llm/checkpoints/mayeec/olmo3-microanneal-round1-olmo3_7b-no-reasoning-anneal-25249a74/step4769'
)


# Moving checkpoints to weka

for model in "${models[@]}"; do
    uv run python -m cookbook.remote gs://$model weka://oe-training-default/$model
done

uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-all-dressed-v18-ce7c0c2a/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-all-dressed-v18-ce7c0c2a/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-all-dressed-v20-1e29004d/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-all-dressed-v20-1e29004d/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-hq-web-baseline-0e513fe6/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-hq-web-baseline-0e513fe6/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-actual-5d35521f/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-actual-5d35521f/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-distill-a2d7cd25/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-distill-a2d7cd25/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-extract-knowledge-76f1ff02/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-extract-knowledge-76f1ff02/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-knowledge_list-de0e6931/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-knowledge_list-de0e6931/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-qa-d2535212/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-qa-d2535212/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-wrap_medium-ab332b66/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-nemotron-hq-synth-wrap_medium-ab332b66/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-olmo3-mix-2b9cd813/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-olmo3-mix-2b9cd813/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-microanneal-round1-olmo3_7b-with-reasoning-anneal-03fb7a8d/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-microanneal-round1-olmo3_7b-with-reasoning-anneal-03fb7a8d/
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo3-mix-1b-5xC-b5ad93d5/step4769  weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo3-mix-1b-5xC-b5ad93d5/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/mayeec/olmo3-microanneal-round1-olmo3_7b-no-reasoning-anneal-25249a74/step4769 weka://oe-training-default/ai2-llm/checkpoints/mayeec/olmo3-microanneal-round1-olmo3_7b-no-reasoning-anneal-25249a74/step4769


# Changing

olmo-cookbook-eval convert "/oe-training-default/ai2-llm/checkpoints/allysone/anneal-round1-100B-olmo3_7b_no-reasoning-anneal-3c193128/step47684" \
 -t olmo-core-v2 \
 --use-beaker \
 --olmo-core-v2-commit-hash  013ef7b54aa2d583f9811ec6211a536da407a4b1 \
 --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
 --huggingface-transformers-commit-hash ca728b8879ce5127ea3e2f8d309c2c5febab5dc5
