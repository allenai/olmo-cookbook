#!/bin/bash

: 'experiments=(
    "olmo3-7b-nanoanneal-1b-with-reasoning-e7224146"
    "olmo3-7b-nanoanneal-1b-hq-web-baseline-25b8e859"
    "olmo3-7b-nanoanneal-1b-no-reasoning-e399aa24"
    "olmo3-7b-nanoanneal-1b-olmo3-mix-4a3170ed"
    "olmo3-7b-nanoanneal-1b-with-reasoning-natural-afc341cf"
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step477 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step477 \
        --allow-dirty \
        --workspace ai2/dolma2
done
'

: 'experiments=(
    "olmo3-7b-microanneal-10b-with-reasoning-natural-3d481ada"
)
for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4769 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4769 \
        --allow-dirty \
        --workspace ai2/dolma2
done'



: 'experiments=(
  'olmo3-7b-nanoanneal-5b-hq-web-baseline-f70d9a14'
  'olmo3-7b-nanoanneal-5b-no-reasoning-feb17b6c'
  'olmo3-7b-nanoanneal-5b-olmo3-mix-d11df886'
  'olmo3-7b-nanoanneal-5b-with-reasoning-d1e64dea'
  'olmo3-7b-nanoanneal-5b-with-reasoning-natural-2886167d'
)
for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step2385 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2385 \
        --allow-dirty \
        --workspace ai2/dolma2
done
'


: 'python -m cookbook.remote \
    gs://ai2-llm/checkpoints/mayeec/olmo3-7b-nanoanneal-5b-dolminos-baseline-88b40b73/step4769 \
    weka://oe-data-default/ai2-llm/checkpoints/mayeec/olmo3-7b-nanoanneal-10b-dolminos-baseline-88b40b73/step4769 \
    --allow-dirty \
    --workspace ai2/dolma2'


: 'experiments=(
  #'olmo3-7b-nanoanneal-1b-dolminos-baseline-4c8c901b/step477'
  #olmo3-7b-web-curve-fitting-1b-d0051eeb/step477
  olmo3-7b-nanoanneal-5b-dolminos-baseline-4fd00746/step2385
  olmo3-7b-web-curve-fitting-2b-4b9afcf0/step954
  olmo3-7b-web-curve-fitting-3b-c13d365c/step1431
  olmo3-7b-web-curve-fitting-4b-ee793de0/step1908
  olmo3-7b-web-curve-fitting-5b-33585590/step2385
  olmo3-7b-web-curve-fitting-6b-50b727ad/step2862
  olmo3-7b-web-curve-fitting-7b-b585a77a/step3338
  olmo3-7b-web-curve-fitting-8b-38cb0258/step3815
  olmo3-7b-web-curve-fitting-9b-e3f85c87/step4292
)
for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id \
        --allow-dirty \
        --workspace ai2/dolma2
done'





: 'experiments=(
  olmo3-7b-web-nemotron-qa-curve-fitting-1b-7ad001ca/step477
  olmo3-7b-web-nemotron-qa-curve-fitting-2b-93ce06e8/step954
  olmo3-7b-web-nemotron-qa-curve-fitting-3b-598e3125/step1431
  olmo3-7b-web-nemotron-qa-curve-fitting-4b-d8d20c9f/step1908
  olmo3-7b-web-nemotron-qa-curve-fitting-5b-5c11a03f/step2385
  olmo3-7b-web-nemotron-qa-curve-fitting-6b-ff2f36ef/step2862
  olmo3-7b-web-nemotron-qa-curve-fitting-7b-90e2e7fd/step3338
  olmo3-7b-web-nemotron-qa-curve-fitting-8b-498be463/step3815
  olmo3-7b-web-nemotron-qa-curve-fitting-9b-7cbf4fb1/step4292
)
for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id \
        --allow-dirty \
        --workspace ai2/dolma2
done'

