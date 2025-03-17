# OLMo Cookbook

OLMost every recipe you need to experiment with the OLMo family of models.

## To Train an OLMo Model

### Prepare your environments

#### Local
1) Install the cookbook CLI

```shell
pip install -e .
```

2) Set up your environment

Optional: (Only if you are using Weka storage for token files)
```shell
  export WEKA_ENDPOINT_URL=<weka-endpoint-url>
  export WEKA_PROFILE=WEKA
```
*Note: Make sure you have WEKA and S3 profiles in your ~/.aws/config and ~/.aws/credentials files.*

#### Remote (Beaker)

1) Create a `Beaker` user account, request access to AI2 clusters, and create an API token.
2) Set up your workspace

```shell
olmo-cookbook prepare-workspace \
  --workspace <workspace> \
  --beaker-token <beaker-token> \
  --aws-config <aws-config> \
  --aws-credentials <aws-credentials> \
  --wandb-api-key <wandb-api-key>
```
*Note: Weka / R2 endpoint urls only need to be set if you are using them for storage.*

If you plan to run jobs on `ai2/augusta-google-1` then your workspace will also require the `Beaker` secrets:
```
GS_INTEROP_KEY
GS_INTEROP_SECRET
```

### Build your training configuration

See `src/cookbook/recipes/train-1b-1xC-dclm.yaml` for an example to clone.

*Note: This cookbook relies on `beaker-py` under the hood and thus requires committing and pushing changes to configuration files before launching a job.*

### Launch your training job

1) `olmo-cookbook launch -c src/cookbook/recipes/train-1b-1xC-dclm.yaml`
2) Follow the interactive prompts. A link to the `Beaker` job will be provided upon successful submission.
3) Monitor your training job in `wandb` or the `Beaker` UI.

## How To Evaluate an OLMo Model

### Convert Checkpoint

#### For models trained with [OLMo](https://github.com/allenai/olmo)

```shell
olmo-cookbook-eval convert \
  -i /oe-training-default/kevinf/checkpoints/OLMo-medium/peteish7-medlr/step477000 \
  -t olmo2 \
  --use-beaker \
  --huggingface-tokenizer allenai/dolma2-tokenizer
```

#### For models trained with [OLMo-core](https://github.com/allenai/olmo-core)

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

Example evaluating a HuggingFace model:

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
