#!/bin/bash

# login to gcloud
# gcloud auth application-default login
# gcloud auth application-default set-quota-project ai2-allennlp
# export GOOGLE_CLOUD_PROJECT=ai2-allennlp


# List of GS paths to process
GS_PATHS=(
    "gs://ai2-llm/checkpoints/stellal/microanneal-noisy-reasoning-olmo3-microanneal-100b-10b-lr8.8e-05-a0f88f73/step4769"
    "gs://ai2-llm/checkpoints/stellal/microanneal-noisy-reasoning-olmo3-microanneal-100b-5b-lr8.8e-05-f3cbf03f/step2385"
)

# Function to convert GS path to weka path

convert_gs_to_weka() {
    local gs_path="$1"
    local prefix="$2"
    
    # Remove gs:// and prepend with the specified prefix
    echo "${gs_path/gs:\/\//${prefix}}"
}

# Process each GS path
for gs_path in "${GS_PATHS[@]}"; do
    echo "Processing: $gs_path"
    
    # Convert paths
    weka_path_remote=$(convert_gs_to_weka "$gs_path" "weka://oe-training-default/")
    weka_path_local=$(convert_gs_to_weka "$gs_path" "/oe-training-default/")
    
    echo "  GS Path: $gs_path"
    echo "  Weka Remote Path: $weka_path_remote"
    echo "  Weka Local Path: $weka_path_local"
    
    
    # Run the second command
    echo "  Running conversion command..."
    if olmo-cookbook-eval convert "$weka_path_local" \
        -t olmo-core-v2 \
        --use-beaker \
        --olmo-core-v2-commit-hash 57a04d0b69047d797c96eede056a211e75b5914a \
        --huggingface-transformers-commit-hash ae3889ced6ed7362e5883671fc6dc4cb4fece5fa; then
        echo "  ✓ Conversion completed successfully"
    else
        echo "  ✗ Conversion failed"
    fi
    
    echo "  Finished processing: $gs_path"
done

echo "All paths processed!"