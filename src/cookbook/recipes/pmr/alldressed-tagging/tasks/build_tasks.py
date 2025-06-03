template = """#!/bin/bash

# Check if /mnt/raid0/models is empty and download artifacts if needed
if [ ! -d "/mnt/raid0/models" ] || [ -z "$(ls -A /mnt/raid0/models)" ]; then
  echo "Models directory is empty, downloading artifacts..."
  mkdir -p "/mnt/raid0/models"
  s5cmd cp -sp s3://ai2-llm/pretraining-data/sources/WebOrganizer/fasttext/models/Topic/may31_lr05_ng3_n3M6_ova_combined-v3.bin /mnt/raid0/models/
  s5cmd cp -sp s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/fasttext_models/oh_uc_wc_eli5_fasttext_model_bigram_200k.bin /mnt/raid0/models/
else
  echo "Models directory already contains files, skipping download..."
fi

SRC_S3_PREFIX="s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3/sa_minlen500/filtered"
DST_S3_PREFIX="s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3/sa_minlen500/filtered/may31_lr05_ng3_n3M6_ova_combined-v3-partitioned"

# Store the input argument
X=XXX

# Step 0: Prepare runtime and local storage
echo "Preparing runtime..."
rm -rf "/mnt/raid0/input"
rm -rf "/mnt/raid0/output"
rm -rf "/mnt/raid0/logs"

mkdir -p "/mnt/raid0/input"
mkdir -p "/mnt/raid0/output"
mkdir -p "/mnt/raid0/logs"

cd ~/datamap-rs
git checkout undfined/tag-alldressed; git pull


# Step 1: Copy from S3 to local storage
echo "Copying data from S3 to local storage..."
s5cmd cp -sp "$SRC_S3_PREFIX/${X}/*" "/mnt/raid0/input/"


# Step 2: Run the tag operation
echo "Running tag operation..."
cargo run --release -- map --input-dir /mnt/raid0/input --output-dir /mnt/raid0/input/annotated/  --config examples/tag_alldressed/tag-docs.yaml  > "/mnt/raid0/tag-docs-${X}.log"


# Step 3: Run the partition operation
echo "Running partition operation..."
cargo run --release -- partition --input-dir /mnt/raid0/input/annotated/step_final/ --output-dir /mnt/raid0/input/partitioned/ --config examples/tag_alldressed/partition-docs.yaml > "/mnt/raid0/logs/partition-docs-${X}.log"


# Step 4: Relocate partitioned files under category directories
echo "Relocating partitioned files..."
OUTPUT_DIR="/mnt/raid0/output/partitioned"
mkdir -p "$OUTPUT_DIR"

# Create directories and move files based on labels
echo "Looking for files matching pattern: /mnt/raid0/input/partitioned/chunk___*__*.jsonl.zst"
found_files=0

for file in /mnt/raid0/input/partitioned/chunk___*__*.jsonl.zst; do
  # Extract the label from the filename
  label=$(basename "$file" | sed 's/chunk___[^_]*__\([^.]*\)\..*/\1/')

  # Fix typo in label electronics_and_hardare
  label=$(echo "$label" | sed 's/electronics_and_hardare/electronics_and_hardware/g')

  # Extract the new filename (remove chunk___*__ prefix)
  new_filename=$(basename "$file" | sed 's/chunk___[^_]*__//')

  # Fix typo in new filename
  new_filename=$(echo "$new_filename" | sed 's/electronics_and_hardare/electronics_and_hardware/g')

  # Create directory if it doesn't exist
  mkdir -p "$OUTPUT_DIR/$label"

  # Move the file
  mv "$file" "$OUTPUT_DIR/$label/$new_filename"

  echo "Moved $file to $OUTPUT_DIR/$label/$new_filename"
done


# Step 5: Copy partitioned files to S3
echo "Copying output to S3..."
s5cmd cp -sp /mnt/raid0/output/partitioned/* "$DST_S3_PREFIX/${X}/"
s5cmd cp -sp "/mnt/raid0/*.log" $DST_S3_PREFIX/logs/

echo "Processing complete for chunk $X"
"""


for i in range(32):
    with open("part_%02d.sh" % i, "w") as f:
        f.write(template.replace("XXX", "%02d" % i))
