#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: mirror_gcs_checkpoints_to_weka.sh --weka-mount MOUNT [OPTIONS] GCS_PARENT_DIR [GCS_PARENT_DIR ...]

Mirrors each step directory (e.g., step0, step1000) under the given GCS checkpoint
parent directories to a Weka path using python -m cookbook.remote.

The destination path is constructed as:
  weka://MOUNT/<gcs-bucket>/<remaining-gcs-path>

Example:
  ./scripts/mirror_gcs_checkpoints_to_weka.sh \
    --weka-mount oe-training-default \
    gs://ai2-llm/checkpoints/ianm/suffix-train-olmo2-5xC-30m-dense-dolma2-100B-baseline-ad869d6a

This will copy (for each step):
  gs://ai2-llm/.../step20000 -> weka://oe-training-default/ai2-llm/.../step20000

Options passed through to cookbook.remote:
  --num-workers INT         Number of workers (default: 10)
  --budget NAME             Beaker budget (default: ai2/oe-base)
  --cluster NAME            Cluster (default: aus)
  --priority LEVEL          Priority (default: high)
  --workspace NAME          Workspace (default: ai2/oe-data)
  --gpus INT                Number of GPUs (default: 0)
  --preemptible             Use preemptible instances
  --allow-dirty             Allow dirty git state
  --dry-run                 Do not submit, print actions
  --local-only              Run copy locally without Beaker

Notes:
  - Requires gsutil in PATH.
  - Uses python -m cookbook.remote under the hood.
USAGE
}

weka_mount=""
num_workers=10
budget="ai2/oe-base"
cluster="aus"
priority="high"
workspace="ai2/oe-data"
gpus=0
preemptible=false
allow_dirty=false
dry_run=false
local_only=false
parents=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weka-mount)
      weka_mount="${2:-}"; shift 2 ;;
    --num-workers)
      num_workers="${2:-}"; shift 2 ;;
    --budget)
      budget="${2:-}"; shift 2 ;;
    --cluster)
      cluster="${2:-}"; shift 2 ;;
    --priority)
      priority="${2:-}"; shift 2 ;;
    --workspace)
      workspace="${2:-}"; shift 2 ;;
    --gpus)
      gpus="${2:-}"; shift 2 ;;
    --preemptible)
      preemptible=true; shift 1 ;;
    --allow-dirty)
      allow_dirty=true; shift 1 ;;
    --dry-run)
      dry_run=true; shift 1 ;;
    --local-only)
      local_only=true; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; while [[ $# -gt 0 ]]; do parents+=("$1"); shift; done ;;
    -*)
      echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *)
      parents+=("$1"); shift ;;
  esac
done

if [[ -z "${weka_mount}" || ${#parents[@]} -eq 0 ]]; then
  usage
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "Error: gsutil not found in PATH" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python not found in PATH" >&2
  exit 1
fi

# Build passthrough args for cookbook.remote
remote_args=(
  --num-workers "${num_workers}"
  --budget "${budget}"
  --cluster "${cluster}"
  --priority "${priority}"
  --workspace "${workspace}"
  --gpus "${gpus}"
)
${preemptible} && remote_args+=(--preemptible) || true
${allow_dirty} && remote_args+=(--allow-dirty) || true
${dry_run} && remote_args+=(--dry-run) || true
${local_only} && remote_args+=(--local-only) || true

for parent in "${parents[@]}"; do
  normalized="${parent%/}/"
  echo "Listing steps under ${normalized} ..."

  # List step directories like .../step0, .../step1000, strip trailing slashes
  steps="$(gsutil ls -d "${normalized}"step*/ 2>/dev/null | sed 's:/*$::' || true)"

  if [[ -z "${steps}" ]]; then
    echo "Warning: no steps found under ${normalized}" >&2
    continue
  fi

  while IFS= read -r step_path; do
    [[ -z "${step_path}" ]] && continue

    if [[ "${step_path}" =~ ^gs://([^/]+)/(.*)$ ]]; then
      gcs_bucket="${BASH_REMATCH[1]}"
      gcs_key="${BASH_REMATCH[2]}"
    else
      echo "Skipping unrecognized GCS path format: ${step_path}" >&2
      continue
    fi

    # Keep trailing slash on both src and dst to avoid prefix collisions downstream
    src_with_slash="${step_path%/}/"
    dst_path="weka://${weka_mount}/${gcs_bucket}/${gcs_key%/}/"
    echo "Mirroring checkpoint: ${step_path} -> ${dst_path}"

    # shellcheck disable=SC2086
    python -m cookbook.remote "${src_with_slash}" "${dst_path}" ${remote_args[*]}
  done <<< "${steps}"
done


