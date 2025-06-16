# How to Tokenize

This is a brief guide on how to tokenize data on EC2.
We will use Poor Man Ray to create a new instance, install Dolma, and then SSH into machine to tokenize the data.

## Step 0: install Poor Man Ray

Clone the OLMo Cookbook repository; install.

```bash
git clone https://github.com/allenai/olmo-cookbook.git
cd olmo-cookbook
pip install -e .
```

Ensure your AWS environment variables are set:
```bash
export AWS_ACCESS_KEY_ID="[your key]"
export AWS_SECRET_ACCESS_KEY="[your secret]"
export AWS_DEFAULT_REGION="us-east-1"
```

## Step 1: create a cluster

Create a cluster on EC2 where we will run tokenization; we will use one `i4i.x32large` instance.

```bash
cluster_name="YOUR_CLUSTER_NAME"
poormanray create -n $cluster_name -t i4i.32xlarge --number 1
```

Then run two setup commands to setup storage and toolkit:

```bash
poormanray setup-d2tk -n $cluster_name  -d
poormanray setup-dolma-python -n $cluster_name -d
```

The `-d` here means do this in the background. You should wait a few minutes to finish. You can check status of first command by running

```bash
poormanray run -n $cluster_name -c 'ls'
```

and check if a `datamap-rs` exists; for the second, run

```bash
poormanray run -n $cluster_name -c 'uv run dolma'
```

and check if a dolma command is found.

## Step 2: Download data to node

Use `list` to get IP of the machine:

```bash
>>> poormanray list -n $cluster_name


Id:     i-xxxxxxxxxxxxxxxxx
Name:   <cluster-name>-0000
Type:   i4i.32xlarge
State:  running
IP:     xxx.yyy.zzz.ttt
Status: 2/2
Tags:   {"Contact": "<your username>", "Name": "<cluster-name>-0000", "Project": "<cluster-name>"}
```

Now SSH into the machine and download the data using `s5cmd`. I recommend doing it inside a tmux session

```bash
ssh ec2-user@xxx.yyy.zzz.ttt

s5cmd cp -sp \
    "s3://ai2-llm/pretraining-data/sources/dataset-name/documents/*" \
    "/mnt/raid0/ai2-llm/pretraining-data/sources/dataset-name/documents"
```

make sure to use "*" at the end of source path, and a trailing / at the end of destination.

## Step 3: Tokenize the  data

Now you can tokenize as follows:

```bash
tokenizer="allenai/dolma2-tokenizer"

uv run huggingface-cli download $tokenizer --local-dir /mnt/raid0/tokenizer

uv run dolma tokens \
    --documents "/mnt/raid0/ai2-llm/pretraining-data/sources/dataset-name/documents/*" \
    --destination "/mnt/raid0/ai2-llm/preprocessed/dataset-name/${tokenizer}" \
    --tokenizer.name_or_path /mnt/raid0/tokenizer/tokenizer.json \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --processes $(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())") \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype uint32
```

## Step 4: Upload data to S3

Finish by uploading the data to S3.

```bash
s5cmd cp -sp \
    "/mnt/raid0/ai2-llm/preprocessed/dataset-name/${tokenizer}/*" \
    "s3://ai2-llm/preprocessed/dataset-name/${tokenizer}/"
```

And then terminate the cluster.

```bash
poormanray terminate -n $cluster_name
```
