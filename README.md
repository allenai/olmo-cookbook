# OLMo Cookbook

OLMost every training recipe you need to perform data interventions with the OLMo family of models.

## How To Train an OLMo Model

### TBD

## How To Evaluate an OLMo Model

### Convert Checkpoint

#### For models trained with old trainer

```shell
olmo-cookbook convert \
  -i /oe-training-default/kevinf/checkpoints/OLMo-medium/peteish7-medlr/step477000 \
  -t olmo2 \
  --use-beaker \
  --huggingface-tokenizer allenai/dolma2-tokenizer
```

#### For models trained with OLMo core

```shell
olmo-cookbook convert \
  -i /oe-training-default/ai2-llm/checkpoints/peteish32-anneal/OLMo2-32Bparams-5Ttokens-100Banneal/step11921 \
  -t olmo-core \
  --use-beaker \
  --huggingface-tokenizer allenai/OLMo-2-1124-7B
```

### Run Evaluation

```shell
olmo-cookbook evaluate -h \
  "/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-seed-42/step23842-hf" \
  --task core:mc --task mmlu:mc --task mmlu:rc --task gen \
  --priority high \
  --cluster aus80g \
  --num-gpus 1 \
  --model-backend vllm \
  --dashboard olmoe-0125
```
