# How To Run OLMo Evals

## Convert Checkpoint

For models trained with old trainer

```shell
python scripts/eval_checkpoint/convert_checkpoint.py \
    -i /oe-training-default/kevinf/checkpoints/OLMo-medium/peteish7-medlr/step477000 \
    -t olmo2 \
    --use-beaker \
    --huggingface-tokenizer allenai/dolma2-tokenizer
```

for models trained with OLMo core

```shell
 python scripts/eval_checkpoint/convert_checkpoint.py \
    -i /oe-training-default/ai2-llm/checkpoints/peteish32-anneal/OLMo2-32Bparams-5Ttokens-100Banneal/step11921 \
    -t olmo-core \
    --use-beaker \
    --huggingface-tokenizer allenai/OLMo-2-1124-7B
```

## Run Evaluation


```shell
python scripts/eval_checkpoint/run_evaluation.py -h \
    "/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-seed-42/step23842-hf" \
    --task core:mc --task mmlu:mc --task mmlu:rc --task gen \
    --priority high \
    --cluster aus80g \
    --num-gpus 1 \
    --model-backend vllm \
    --dashboard olmoe-0125
```
