#!/bin/bash

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file>"
    echo "Input file should contain S3 paths (s3://...) separated by newlines"
    exit 1
fi

input_file="$1"
output_cmd_file="tokenize_commands.sh"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found"
    exit 1
fi

tokenizer="allenai/dolma2-tokenizer"

uv run huggingface-cli download $tokenizer --local-dir tokenizer

# Clear or create the output command file
echo "#!/bin/bash" > "$output_cmd_file"

# Read each line from the input file
while IFS= read -r line; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi
    
    # Remove s3:// prefix
    path="${line#s3://}"
    
    echo "Processing path: $path"
    
    # Build the tokenization command (formatted for readability)
    cmd="uv run dolma tokens \\
        --documents \"/mnt/raid0/${path}*\" \\
        --destination \"/mnt/raid0/${path}${tokenizer}\" \\
        --tokenizer.name_or_path tokenizer/tokenizer.json \\
        --tokenizer.eos_token_id 100257 \\
        --tokenizer.pad_token_id 100277 \\
        --no-tokenizer.segment_before_tokenization \\
        --tokenizer.encode_special_tokens \\
        --processes \$(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())') \\
        --max_size 4_000_000_000 \\
        --sample_ring_prop \\
        --dtype uint32"
    
    # Append the command to the output file
    echo "$cmd" >> "$output_cmd_file"
    
    
    
done < "$input_file"

echo "Finished processing all paths"
echo "All commands have been saved to $output_cmd_file"
