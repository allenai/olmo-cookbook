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

## Step 2: Connect to machine and setup

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
```

## Step 3: Download data to node

Write a line separated file of the paths you want to tokenize. These can be s3 paths to a specific object, to a prefix containing jsonl files, or to a prefix with a single level of subdirectories (e.g. different domains) which then contain jsonl files.
```bash
cat <<EOF > s3_paths.txt
s3://path1/to/data/
s3://path2/to/specific/file.jsonl.gz
EOF
```

download cookbook
```bash
https://github.com/allenai/olmo-cookbook.git
```

Use the orchestrator to download (NOTE: all tokenizer_orchestrator commands must be run from ~ so that dolma is available)
```bash
uv run python olmo-cookbook/scripts/tokenizer_orchestrator.py download s3_paths.txt
```


## Step 4: Tokenize the  data

Now you can tokenize as follows:

```bash
uv run python olmo-cookbook/scripts/tokenizer_orchestrator.py tokenize s3_paths.txt
```

You may need to trouble shoot some failed tokenization jobs (issues such as missing IDs or metadata files mixed with jsonl). You can use the following command to look for the bad outputs from these:
```bash
find /mnt/raid0/ai2-llm/preprocessed/ -type f -exec du -h {} + | awk '$1=="4.0K"' | less
```

## Step 4: Upload data to S3


This needs gcloud installed:
```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-456.0.0-linux-x86_64.tar.gz && tar -xf google-cloud-cli-456.0.0-linux-x86_64.tar.gz && ./google-cloud-sdk/install.sh -q
gcloud auth login
gcloud config set project ai2-allennlp
```

Finish by uploading the data to S3 and GS


```bash
uv run python olmo-cookbook/scripts/tokenizer_orchestrator.py upload s3_paths.txt
```

And then terminate the cluster.

## Step 5: Get your paths to use in a config

```bash
uv run python olmo-cookbook/scripts/tokenizer_orchestrator.py destination_paths s3_paths.txt
```

## Step 5: Clean up

```bash
poormanray terminate -n $cluster_name
```
