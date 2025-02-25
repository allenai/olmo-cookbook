# OLMo Cookbook

OLMost every recipe you need to perform data interventions with the OLMo family of models.

## How To Train an OLMo Model

### Build your training configuration

See src/cookbook/recipes/train-1b-1xC-dclm.yaml for an example to clone

### Launch your training job

1) `olmo-cookbook launch -c src/cookbook/recipes/train-1b-1xC-dclm.yaml`

2) Follow the interactive prompts. A link to the Beaker job will be provided upon successful submission.

3) Monitor your training job in wandb or Beaker

## How To Evaluate an OLMo Model

### Convert Checkpoint

#### For models trained with old trainer

```shell
olmo-cookbook-eval convert \
  -i /oe-training-default/kevinf/checkpoints/OLMo-medium/peteish7-medlr/step477000 \
  -t olmo2 \
  --use-beaker \
  --huggingface-tokenizer allenai/dolma2-tokenizer
```

#### For models trained with OLMo core

```shell
olmo-cookbook-eval convert \
  "/oe-training-default/ai2-llm/checkpoints/peteish32-anneal/OLMo2-32Bparams-5Ttokens-100Banneal/step11921" \
  -t olmo-core \
  --use-beaker \
  --huggingface-tokenizer allenai/OLMo-2-1124-7B
```

### Run Evaluation

```shell
olmo-cookbook-eval evaluate \
  "/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-seed-42/step23842-hf" \
  --tasks core:mc --tasks mmlu:mc --tasks mmlu:rc --tasks gen \
  --priority high \
  --cluster aus80g \
  --num-gpus 1 \
  --model-backend vllm \
  --dashboard olmoe-0125
```

Another example is running evaluation for a Hugging Face model

```shell
olmo-cookbook-eval evaluate \
  mistralai/Mistral-Small-24B-Base-2501
  --tasks gen-no-jp
  --priority high
  --cluster aus80g
  --num-gpus 1
  --model-backend vllm
  --dashboard peteish32
```
