uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v16-81576ed2/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v16-81576ed2/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v17-ecbe51d5/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v17-ecbe51d5/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v18-8cb39ff8/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v18-8cb39ff8/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v20-10ddb0f7/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v20-10ddb0f7/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-hq-web-baseline-d49bb1b5/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-hq-web-baseline-d49bb1b5/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-actual-869bbd11/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-actual-869bbd11/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-distill-21a1e733/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-distill-21a1e733/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-extract-knowledge-0017cc3b/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-extract-knowledge-0017cc3b/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-knowledge_list-9e85c7de/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-knowledge_list-9e85c7de/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-qa-12705e4b/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-qa-12705e4b/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-wrap_medium-f3548f29/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-wrap_medium-f3548f29/step4769
uv run python -m cookbook.remote gs://ai2-llm/checkpoints/lucas/olmo2-microanneal-olmo3-mix-8fa583d0/step4769 weka://oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-olmo3-mix-8fa583d0/step4769

####


uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v16-81576ed2/step4769' -t olmo-core-v2 --use-beaker

uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v17-ecbe51d5/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v18-8cb39ff8/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-all-dressed-v20-10ddb0f7/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-hq-web-baseline-d49bb1b5/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-actual-869bbd11/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-distill-21a1e733/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-extract-knowledge-0017cc3b/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-knowledge_list-9e85c7de/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-qa-12705e4b/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-nemotron-hq-synth-wrap_medium-f3548f29/step4769' -t olmo-core-v2 --use-beaker
uv run olmo-cookbook-eval convert '/oe-training-default/ai2-llm/checkpoints/lucas/olmo2-microanneal-olmo3-mix-8fa583d0/step4769' -t olmo-core-v2 --use-beaker


######

# 'olmo2-microanneal-nemotron-hq-synth-extract-knowledge-0017cc3b'
checkpoints=(
    'olmo2-microanneal-all-dressed-v16-81576ed2'
    'olmo2-microanneal-all-dressed-v17-ecbe51d5'
    'olmo2-microanneal-all-dressed-v18-8cb39ff8'
    'olmo2-microanneal-all-dressed-v20-10ddb0f7'
    'olmo2-microanneal-hq-web-baseline-d49bb1b5'
    'olmo2-microanneal-nemotron-hq-actual-869bbd11'
    'olmo2-microanneal-nemotron-hq-synth-distill-21a1e733'
    'olmo2-microanneal-nemotron-hq-synth-knowledge_list-9e85c7de'
    'olmo2-microanneal-nemotron-hq-synth-qa-12705e4b'
    'olmo2-microanneal-nemotron-hq-synth-wrap_medium-f3548f29'
    'olmo2-microanneal-olmo3-mix-8fa583d0'
)

for checkpoint in "${checkpoints[@]}"; do
    uv run olmo-cookbook-eval evaluate \
        "/oe-training-default/ai2-llm/checkpoints/lucas/${checkpoint}/step4769-hf" \
    --tasks olmo3:dev:7b:main \
    --priority urgent \
    --cluster aus80g \
    --num-gpus 1 \
    --partition-size 8 \
    --model-backend vllm \
    --dashboard 'olmo3-midtraining-web'
done
