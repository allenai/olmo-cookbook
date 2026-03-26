# Web Cascade Tokenization

Documentation for tokenizing the `cascade_surviving_docs.jsonl` dataset.

## Dataset Summary

| Field | Value |
|-------|-------|
| Source File | `/Users/kylel/ai2/web-poison/comments/data/cascade_61k/cascade_surviving_docs.jsonl` |
| Size | 1.4 GB |
| Documents | ~60,297 |
| Text Field | `extracted_text` (non-standard) |
| ID Field | `url` (no `id` field exists) |

## Output

| Field | Value |
|-------|-------|
| S3 Location | `s3://ai2-llm/preprocessed/web-poison/cascade_61k/` |
| Token File | `part-0-00000.npy` (1.17 GB) |
| Metadata File | `part-0-00000.csv.gz` (2.5 MB) |
| Total Tokens | 292M |
| Tokenizer | `allenai/dolma2-tokenizer` |

## Tokenization Steps

### 1. Create EC2 Cluster

```bash
cluster_name="kyle-cascade-tokenize"
poormanray create -n $cluster_name -t i4i.32xlarge --number 1
```

Instance created: `i-0772d2f072a68607b` (`kyle-cascade-tokenize-0000`)

### 2. Setup Cluster

Install d2tk (datamap toolkit) and dolma-python:

```bash
poormanray setup-d2tk -n $cluster_name -d
poormanray setup-dolma-python -n $cluster_name -d
```

### 3. Create Data Directories

```bash
poormanray run -n $cluster_name -c 'sudo mkdir -p /mnt/raid0/data /mnt/raid0/output && sudo chown -R ec2-user:ec2-user /mnt/raid0'
```

### 4. Upload Dataset

Get instance IP:
```bash
poormanray list -n $cluster_name
# IP: 35.153.144.163
```

Add host key and upload:
```bash
ssh-keyscan -H 35.153.144.163 >> ~/.ssh/known_hosts
scp /Users/kylel/ai2/web-poison/comments/data/cascade_61k/cascade_surviving_docs.jsonl \
    ec2-user@35.153.144.163:/mnt/raid0/data/
```

### 5. Download Tokenizer

```bash
poormanray run -n $cluster_name -c 'uv run huggingface-cli download allenai/dolma2-tokenizer --local-dir /mnt/raid0/tokenizer'
```

### 6. Run Tokenization

```bash
poormanray run -n $cluster_name -c 'uv run dolma tokens \
    --documents "/mnt/raid0/data/cascade_surviving_docs.jsonl" \
    --destination "/mnt/raid0/output/cascade_61k_tokens" \
    --tokenizer.name_or_path /mnt/raid0/tokenizer/tokenizer.json \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --fields.text_field_name extracted_text \
    --fields.id_field_name url \
    --ring_size 8 \
    --processes $(nproc) \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype uint32'
```

**Key flags for non-standard fields:**
- `--fields.text_field_name extracted_text` - Use `extracted_text` instead of default `text`
- `--fields.id_field_name url` - Use `url` instead of default `id`

### 7. Upload to S3

```bash
poormanray run -n $cluster_name -c 's5cmd cp -sp "/mnt/raid0/output/cascade_61k_tokens/*" "s3://ai2-llm/preprocessed/web-poison/cascade_61k/"'
```

### 8. Terminate Cluster

```bash
poormanray terminate -n $cluster_name
```

## Final Statistics

```
Documents: 60,297
Tokens:    292,000,000 (292M)
Files:     1
```

## Verification

Check S3 contents:
```bash
aws s3 ls s3://ai2-llm/preprocessed/web-poison/cascade_61k/
# 2026-03-25 11:23:06    2580719 part-0-00000.csv.gz
# 2026-03-25 11:23:06 1166514820 part-0-00000.npy
```

## Date

Tokenization completed: 2026-03-25
