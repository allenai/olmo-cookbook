#!/bin/bash

: '
Usage:
  ./gantry_soup_checkpoints.sh -c [CHECKPOINT_PATH_1] -c [CHECKPOINT_PATH_2] ... -c [CHECKPOINT_PATH_N] -d [OUTPUT_DIR]

Arguments:
  -c, --checkpoint    (required) Path to each checkpoint.
  -d, --output-dir     (required) Path to output directory.

Example:
  # Run with 4 checkpoints
  ./gantry_soup_checkpoints.sh \
  -c /path/to/checkpoint_1 \
  -c /path/to/checkpoint_2 \
  -c /path/to/checkpoint_3 \
  -c /path/to/checkpoint_4 \
  -d /path/to/output_dir
'


usage() {
  echo "Usage: $0 -c [CHECKPOINT_PATH_1] -c [CHECKPOINT_PATH_2] ... -c [CHECKPOINT_PATH_N] -d [OUTPUT_DIR]"
  exit 1
}

# Initialize arrays for checkpoints
CHECKPOINTS=()
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--checkpoint)
      CHECKPOINTS+=("$2")
      shift 2
      ;;
    -d|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate required arguments
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "Error: At least one checkpoint path is required."
  usage
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Output directory is required."
  usage
fi

# Display parsed arguments
echo "Checkpoints to soup:"
for checkpoint in "${CHECKPOINTS[@]}"; do
  echo "  - $checkpoint"
done
echo "Output directory: $OUTPUT_DIR"



# Check if gantry is available
if command -v gantry &> /dev/null; then
  GANTRY_CMD="gantry"
elif command -v uv &> /dev/null && uv run gantry --help &> /dev/null; then
  GANTRY_CMD="uv run gantry"
else
  echo "Error: gantry command not found. Please install gantry or ensure it's available via uv."
  exit 1
fi


set -ex

${GANTRY_CMD} run \
  --description "Evaluating ${MODEL_NAME_OR_PATH} on HELMET" \
  --allow-dirty \
  --cluster "${CLUSTER}" \
  --gpus "${NUM_GPUS}" \
  --priority "${PRIORITY}" \
  --weka=oe-training-default:/weka/oe-training-default \
  --workspace "${WORKSPACE}" \
  --budget ai2/oe-base \
  ${SECRET_FLAG} \
  --results "/results" \
  --yes \
  --beaker-image ai2/cuda12.8-dev-ubuntu22.04-notorch \
  --install "
    export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a6d795d593046abd490b16349bcd9b40feedd334/vllm-0.9.2rc2.dev58%2Bga6d795d59-cp38-abi3-manylinux1_x86_64.whl; \
    export VLLM_USE_PRECOMPILED=1; \
    apt install -y protobuf-compiler && \
    uv venv --python 3.12 && \
    uv pip install pyyaml -r requirements.txt && \
    git clone ${TRANSFORMERS_URL} transformers_olmo3 && \
    cd transformers_olmo3 && \
    git checkout ${TRANSFORMERS_COMMIT} && \
    cd .. && \
    uv pip install -e transformers_olmo3 && \
    git clone ${VLLM_URL} vllm_olmo3 && \
    cd vllm_olmo3 && \
    git checkout ${VLLM_COMMIT} && \
    cd .. && \
    uv pip install -e vllm_olmo3 && \
    uv pip install flash-attn --no-build-isolation
  " \
  --no-conda \
  --timeout ${TIMEOUT} \
  -- bash -c "export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 && export MAX_LENGTH='${MAX_LENGTH}' && export OUTPUT_DIR='${OUTPUT_DIR}' && export MODEL_NAME_OR_PATH='${MODEL_NAME_OR_PATH}' && export NUM_GPUS='${NUM_GPUS}' && bash run_and_evaluate.sh ${OPTIONS} && cp -r ${OUTPUT_DIR}/* /results/"
