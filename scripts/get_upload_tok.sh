#!/bin/bash

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file>"
    echo "Input file should contain S3 paths (s3://...) separated by newlines"
    exit 1
fi

input_file="$1"
output_file="s5cmd_commands.sh"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found"
    exit 1
fi

tokenizer="allenai/dolma2-tokenizer"

# Clear output file
echo "set -e" > "$output_file"

# Read each line from the input file
while IFS= read -r line; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi

    # Remove s3:// prefix
    path="${line#s3://}"

    # Extract the path to data by removing the prefix and suffix
    # Remove "ai2-llm/pretraining-data/sources/" from the beginning
    path_to_data="${path#ai2-llm/pretraining-data/sources/}"

    # Remove the tokenizer suffix if it exists
    path_to_data="${path_to_data%${tokenizer}/}"
    path_to_data="${path_to_data%/}"

    dest="s3://ai2-llm/preprocessed/${path_to_data}/${tokenizer}/"

    # echo "s5cmd rm \"${dest}\"" >> "$output_file"
    echo "s5cmd cp -sp \"/mnt/raid0/${path}${tokenizer}/\" \"${dest}\"" >> "$output_file"
    echo "" >> "$output_file"

done < "$input_file"

echo "Finished writing s5cmd commands to $output_file"
