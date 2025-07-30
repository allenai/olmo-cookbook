# Tokenization Orchestrator Usage Examples

## Basic Usage

The `tokenization_orchestrator.py` script replaces the functionality of the previous messy scripts with a clean, unified interface.

### Commands

1. **Download data from remote storage:**
   ```bash
   python scripts/tokenization_orchestrator.py download mix_lists/input_file_no_sub_dir
   ```

2. **Tokenize downloaded data:**
   ```bash
   python scripts/tokenization_orchestrator.py tokenize mix_lists/input_file_no_sub_dir
   ```

3. **Upload tokenized data:**
   ```bash
   python scripts/tokenization_orchestrator.py upload mix_lists/input_file_no_sub_dir
   ```

4. **Get destination paths:**
   ```bash
   python scripts/tokenization_orchestrator.py destination_paths mix_lists/input_file_no_sub_dir
   ```

### Custom Configuration

You can customize the behavior with command-line options:

```bash
python scripts/tokenization_orchestrator.py download input_file.txt \
  --remote-prefix "gs://" \
  --input-prefix "my-bucket/raw-data/" \
  --output-prefix "my-bucket/processed/" \
  --local-dir "/tmp/processing" \
  --tokenizer "custom/tokenizer"
```

### Full Pipeline Example

Process a complete pipeline:

```bash
# Download
python scripts/tokenization_orchestrator.py download mix_lists/input_file_no_sub_dir

# Tokenize 
python scripts/tokenization_orchestrator.py tokenize mix_lists/input_file_no_sub_dir

# Upload
python scripts/tokenization_orchestrator.py upload mix_lists/input_file_no_sub_dir

# Get final paths
python scripts/tokenization_orchestrator.py destination_paths mix_lists/input_file_no_sub_dir
```

## Key Features

- **Error Reporting**: Each command reports errors at the end, no silent failures
- **Empty File Removal**: Download command automatically detects and removes empty JSONL files
- **Automatic ID Field Detection**: Handles different ID field names and types automatically
- **Disk Space Checking**: Ensures 10TB+ free space before tokenization
- **Subdirectory Handling**: Automatically detects and handles directory structures
- **Clean Output**: Provides clear progress and final destination paths with wildcards

## Input File Format

The input file should contain one S3/GS path per line:
```
s3://bucket/path/to/data1/
s3://bucket/path/to/data2.jsonl.gz
gs://another-bucket/path/to/data3/
```

Lines starting with `#` are treated as comments and ignored.
