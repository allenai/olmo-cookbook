#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_gcs_evals.sh --dashboard NAME --hf-token TOKEN WEKA_PARENT_DIR [WEKA_PARENT_DIR ...]

Runs olmo-cookbook evals for each step directory (e.g., step0, step1000) under the given
Weka checkpoint parent directories.

Example:
  ./scripts/run_gcs_evals.sh \
    --dashboard my-evals \
    --hf-token "$HF_TOKEN" \
    weka://oe-training-default/ai2-llm/checkpoints/ianm/suffix-train-olmo2-5xC-30m-dense-falcon-1fd53d15 \
    weka://oe-training-default/ai2-llm/checkpoints/ianm/suffix-train-olmo2-5xC-30m-dense-dolma2-100B-baseline-ad869d6a
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

if ! command -v gsutil >/dev/null 2>&1; then
  echo "Error: gsutil not found in PATH" >&2
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

  # Derive the matching GCS prefix from the weka path: weka://MOUNT/<bucket>/<key...>
  bucket="${rel%%/*}"
  key="${rel#${bucket}/}"
  gs_parent="gs://${bucket}/${key%/}"

  echo "Listing steps under ${gs_parent} ..."

  # List step directories on GCS and strip trailing slashes
  gs_steps="$(gsutil ls -d "${gs_parent%/}"/step*/ 2>/dev/null | sed 's:/*$::' | sort || true)"

  if [[ -z "${gs_steps}" ]]; then
    echo "Warning: no steps found under ${gs_parent}" >&2
    continue
  fi

  # Iterate GCS steps, evaluating only the allowed step names
  while IFS= read -r gs_step; do
    [[ -z "${gs_step}" ]] && continue
    step_name="${gs_step##*/}"
    # Check if this step is in the allowed list
    is_allowed=false
    for s in "${allowed_steps[@]}"; do
      if [[ "${step_name}" == "${s}" ]]; then
        is_allowed=true
        break
      fi
    done
    [[ "${is_allowed}" != true ]] && continue
    ## olmo3:dev:1b:bpb
    local_step_path="${local_parent%/}/${step_name}"
    echo "Evaluating checkpoint: ${local_step_path}"
    olmo-cookbook-eval evaluate "${local_step_path}" \
      --tasks olmo3:dev:1b:main \
      --priority normal \
      --cluster aus80g \
      --partition-size 8 \
      --num-gpus 1 \
      --beaker-retries 3 \
      --model-backend olmo_core \
      --fim-tokens l2c \
      --dashboard "${dashboard}" \
      --workspace ai2/olmo-3-microanneals \
      --huggingface-secret "${hf_token}" \
      --use-hf-token
  done <<< "${gs_steps}"
done


