#!/bin/bash

# Script to download Hugging Face models with specific revisions
# Usage: ./download_models.sh <output-dir>

set -e

# Check if output directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <output-dir>"
    echo "Example: $0 ./models"
    exit 1
fi

OUTPUT_DIR="$1"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define models and their revisions
declare -A models_revisions=(
    ["allenai/OLMo-2-0425-1B"]="stage2-ingredient3-step20000-tokens42B stage2-ingredient3-step21000-tokens45B stage2-ingredient3-step22000-tokens47B stage2-ingredient3-step23000-tokens49B stage2-ingredient3-step23852-tokens51B"
    ["allenai/OLMo-2-1124-7B"]="stage2-ingredient3-step8000-tokens34B stage2-ingredient3-step9000-tokens38B stage2-ingredient3-step10000-tokens42B stage2-ingredient3-step11000-tokens47B stage2-ingredient3-step11931-tokens50B"
    ["allenai/OLMo-2-1124-13B"]="stage2-ingredient4-step32000-tokens269B stage2-ingredient4-step33000-tokens277B stage2-ingredient4-step34000-tokens286B stage2-ingredient4-step35000-tokens294B stage2-ingredient4-step35773-tokens300B"
)

# Function to download a model with a specific revision
download_model() {
    local repo="$1"
    local revision="$2"
    local output_dir="$3"
    
    # Extract owner and model name from repo
    local owner=$(echo "$repo" | cut -d'/' -f1)
    local model_name=$(echo "$repo" | cut -d'/' -f2)
    
    # Create directory name in the format: owner--model-name--revision-name
    local dir_name="${owner}--${model_name}--${revision}"
    local target_dir="${output_dir}/${dir_name}"
    
    # Skip download if the target directory already exists
    if [ -d "$target_dir" ]; then
        echo "Skipping ${repo}@${revision}, already exists at ${target_dir}"
        return
    fi
    
    echo "Downloading ${repo} at revision ${revision} to ${target_dir}..."
    
    # Use huggingface-hub CLI to download
    huggingface-cli download \
        --repo-type model \
        --revision "$revision" \
        --local-dir "$target_dir" \
        --local-dir-use-symlinks False \
        "$repo"
    
    echo "✓ Downloaded ${repo}@${revision} to ${target_dir}"
}

# Download all models and revisions
echo "Starting model downloads to: $OUTPUT_DIR"
echo "======================================="

for repo in "${!models_revisions[@]}"; do
    echo ""
    echo "Processing repository: $repo"
    echo "----------------------------"
    
    # Split revisions by space and iterate
    for revision in ${models_revisions[$repo]}; do
        download_model "$repo" "$revision" "$OUTPUT_DIR"
    done
done

echo ""
echo "======================================="
echo "All downloads completed!"
echo "Models downloaded to: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
ls -la "$OUTPUT_DIR"