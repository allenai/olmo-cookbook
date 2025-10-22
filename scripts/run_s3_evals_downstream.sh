#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_s3_evals_downstream.sh --dashboard NAME --hf-token TOKEN WEKA_PARENT_DIR [WEKA_PARENT_DIR ...]

Runs olmo-cookbook downstream evals for each allowed step directory (e.g., step5000, step10000)
under the given Weka checkpoint parent directories, but expects S3-backed paths (mirrored on S3).

Example:
  ./scripts/run_s3_evals_downstream.sh \
    --dashboard my-evals \
    --hf-token "$HF_TOKEN" \
    weka://oe-training-default/s3-bucket-name/checkpoints/some-experiment \
    weka://oe-training-default/another-s3-bucket/checkpoints/other-experiment
USAGE
}

dashboard=""
hf_token=""
parents=()
allowed_steps=("step5000" "step10000" "step15000" "step20000") # "step22204"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dashboard)
      dashboard="${2:-}"; shift 2 ;;
    -t|--hf-token)
      hf_token="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    -*)
      echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *)
      parents+=("$1"); shift ;;
  esac
done

if [[ -z "${dashboard}" || -z "${hf_token}" || ${#parents[@]} -eq 0 ]]; then
  usage
  exit 1
fi

if ! command -v olmo-cookbook-eval >/dev/null 2>&1; then
  echo "Error: olmo-cookbook-eval not found in PATH" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws (cli) not found in PATH" >&2
  exit 1
fi

for parent in "${parents[@]}"; do
  # Normalize input: accept either weka://MOUNT/path... or /MOUNT/path...
  if [[ "${parent}" =~ ^weka://([^/]+)/?(.*)$ ]]; then
    mount="${BASH_REMATCH[1]}"
    rel="${BASH_REMATCH[2]}"
    local_parent="/${mount}/${rel}"
    weka_prefix="weka://${mount}/${rel%/}"
  else
    # Treat as local path like /oe-training-default/ai2-llm/...
    local_parent="${parent%/}"
    # Extract mount from first path segment
    tmp="${local_parent#/}"
    mount="${tmp%%/*}"
    rel="${local_parent#/${mount}/}"
    weka_prefix="weka://${mount}/${rel%/}"
  fi

  # Derive the matching S3 prefix from the weka path: weka://MOUNT/<bucket>/<key...>
  bucket="${rel%%/*}"
  key="${rel#${bucket}/}"
  s3_parent="s3://${bucket}/${key%/}"

  echo "Listing steps under ${s3_parent} ..."

  # List step directories on S3 and strip trailing slashes
  # Require AWS CLI v2
  s3_steps="$(aws s3 ls "${s3_parent%/}/" 2>/dev/null | awk '/PRE step/ { print $2 }' | sed 's:/*$::' | sort || true)"

  if [[ -z "${s3_steps}" ]]; then
    echo "Warning: no steps found under ${s3_parent}" >&2
    continue
  fi

  # Iterate each S3 step and evaluate only if in the allowed list
  while IFS= read -r step_name; do
    [[ -z "${step_name}" ]] && continue
    step_name="${step_name%%/}"
    # Check if this step is in the allowed list
    is_allowed=false
    for s in "${allowed_steps[@]}"; do
      if [[ "${step_name}" == "${s}" ]]; then
        is_allowed=true
        break
      fi
    done
    [[ "${is_allowed}" != true ]] && continue

    s3_step_path="${s3_parent%/}/${step_name}"
    echo "Evaluating checkpoint: ${s3_step_path}"
    olmo-cookbook-eval evaluate "${s3_step_path}" \
      --tasks mmlu:bpb \
      --priority low \
      --cluster ai2/titan \
      --partition-size 8 \
      --num-gpus 1 \
      --beaker-retries 3 \
      --model-backend olmo_core \
      --fim-tokens l2c \
      --dashboard "${dashboard}" \
      --workspace ai2/olmo-3-microanneals \
      --huggingface-secret "${hf_token}" \
      --use-hf-token
  done <<< "${s3_steps}"
done



