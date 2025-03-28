# OLMo Cookbook

OLMost every recipe you need to experiment with the OLMo family of models.

## How To Train an OLMo Model

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

1) Create a `Beaker` user account, request access to AI2 clusters, and create a Beaker user token.
2) Set up your workspace

```shell
olmo-cookbook prepare-user-workspace \
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

## Running OLMo-core script

You can launch any OLMo core training script using the cookbook.
By default, any script in [src/scripts/train](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train) can be launched.

Here's an example of how to train a 1B model for 50B tokens on 16 GPUs on the `ai2/augusta-google-1` cluster.

```shell
olmo-cookbook-core launch \
  -d dolmino50 \
  -m OLMo2-1B \
  -n 50e9T \
  -i petew/olmo-core-tch260cu126-v2.0.1 \
  -p urgent \
  -c ai2/augusta-google-1 \
  -g 16
```

Let's break down the command:

- `-d dolmino50`: The data mix to use for training. This data mix is at [data/mixes/dolmino50.yaml](src/cookbook/data/mixes), but you can use any path to a data mix file (i.e., a plain text file with a list on npy tokens files)
- `-m OLMo2-1B`: The model to train. This is the configuration [src/scripts/train/OLMo2-1B.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/train/OLMo2-1B.py). You can also provide a path to any training script written in OLMo-core.
- `-n 50e9T`: The number of tokens to train on (50B tokens).
- `-i petew/olmo-core-tch260cu126-v2.0.1`: The image to use for training.
- `-p urgent`: The priority of the job.
- `-c ai2/augusta-google-1`: The cluster to use for training.
- `-g 16`: The number of GPUs to use for training.

Use the `--dry-run` flag to print the command without launching the job; to view all available flags, run `olmo-cookbook-core launch --help`.

At the moment, we pin OLMo-core to commit [`2f66fd9`](https://github.com/allenai/OLMo-core/tree/2f66fd95c17c9779be9930f8fb80803293c2dc30), but you can override this by setting the `--olmo-core-commit-hash` flag.


## EC2 CLI

The EC2 CLI is a tool for managing EC2 instances. We will describe its use by example.

First, you want to install the cookbook CLI.

```shell
pip install -e .
```

Then, you can create a cluster of instances; by default, instances will be `i4i.xlarge` and will be tagged with the project name and owner; they will use the `us-east-1` region and use your SSH key at `~/.ssh/id_rsa`.

Let's say you wanna create a cluster named `chipstest`:

```shell
olmo-cookbook-ec2 create --name chipstest --number 5 --instance i4i.2xlarge --detach
```

This will create 5 instances as part of a cluster with the name `chipstest`; the `--detach` flag means that the
process will return immediately and the instances will be created in the background.

You can check the status of the instances by listing them:

```shell
olmo-cookbook-ec2 list --name chipstest
```

After the instances are create, you wanna set up AWS credentials and D2TK pipeline on them. You can do this by running the following command:

```shell
olmo-cookbook-ec2 setup-d2tk --name chipstest
```

To run a command on all instances in the cluster, you can use the following command:

```shell
olmo-cookbook-ec2 run --name chipstest --command "echo 'Hello, world!'"
```

But, most likely you wanna queue a bunch of jobs to run on the instances. You can do this by creating a directory with as many bash scripts as job units, and then running the following command:

```shell
olmo-cookbook-ec2 map --name chipstest --scripts-dir tmp/test_scripts
```

This will run all the scripts in the `tmp/test_scripts` directory on all the instances in the cluster.

Once you are done with the jobs, you can terminate the cluster:

```shell
olmo-cookbook-ec2 terminate --name chipstest
```

This will terminate all the instances in the cluster and delete the cluster.
