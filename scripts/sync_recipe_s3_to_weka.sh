#!/usr/bin/env bash

set -euo pipefail

# Sync all active s3 dataset paths in a cookbook recipe YAML to the local Weka mount.
# Example mapping: s3://bucket/path/to/data/* -> /weka/bucket/path/to/data/*
#
# Usage:
#   scripts/sync_recipe_s3_to_weka.sh path/to/recipe.yaml [/weka]
#
# Notes:
# - Requires s5cmd in PATH and AWS creds configured for S3 access.
# - Uses `s5cmd run` with `cp --parents` so directory structure is preserved under DEST_ROOT.
# - Only non-commented YAML list entries like `- s3://...` are considered.

RECIPE_PATH=${1:-}
DEST_ROOT=${2:-/weka}

if [[ -z "${RECIPE_PATH}" ]]; then
  echo "Usage: $0 path/to/recipe.yaml [/weka]" >&2
  exit 1
fi

if [[ ! -f "${RECIPE_PATH}" ]]; then
  echo "Recipe not found: ${RECIPE_PATH}" >&2
  exit 1
fi

if ! command -v s5cmd >/dev/null 2>&1; then
  echo "s5cmd not found in PATH. Please install s5cmd." >&2
  exit 1
fi

# Ensure destination root exists (mount point should already be present)
if [[ ! -d "${DEST_ROOT}" ]]; then
  echo "Destination root does not exist: ${DEST_ROOT}" >&2
  exit 1
fi

# Extract active s3/weka source paths from YAML
# - Strip full-line comments and inline comments after '#'
# - Match URLs anywhere on the line
mapfile -t SRC_PATHS < <(sed -E 's/#.*$//' "${RECIPE_PATH}" \
  | awk '{ for (i=1;i<=NF;i++) if ($i ~ /^(s3|weka):\/\//) print $i }' \
  | sed -E 's/[",]$//' \
  | sort -u)

if [[ ${#SRC_PATHS[@]} -eq 0 ]]; then
  echo "No active s3/weka paths found in recipe: ${RECIPE_PATH}" >&2
  exit 1
fi

echo "Found ${#SRC_PATHS[@]} path patterns in ${RECIPE_PATH}." >&2
echo "Destination root: ${DEST_ROOT}" >&2

# Build s5cmd runfile: sync selected objects while preserving full key path under DEST_ROOT
TMP_RUNFILE=$(mktemp)
trap 'rm -f "${TMP_RUNFILE}"' EXIT

for p in "${SRC_PATHS[@]}"; do
  # Normalize to s3:// for source (convert weka://<mount>/<bucket>/... -> s3://<bucket>/...)
  if [[ "${p}" == weka://* ]]; then
    tmp=${p#weka://}         # <mount>/<bucket>/key...
    tmp=${tmp#*/}            # <bucket>/key...
    p_s3="s3://${tmp}"
  else
    p_s3="${p}"
  fi

  # Parse bucket and key pattern
  bucket=${p_s3#s3://}; bucket=${bucket%%/*}
  key_with_glob=${p_s3#s3://${bucket}/}

  # Compute the longest non-glob prefix (no *, ?, [ )
  prefix=$(printf '%s' "${key_with_glob}" | sed -E 's/[\*\?\[].*$//')
  rest=${key_with_glob#${prefix}}

  # If there were no globs, back up to the directory containing the file
  if [[ -z "${rest}" ]]; then
    prefix=${prefix%/*}/
  fi

  # Ensure prefix ends with a single slash if non-empty
  if [[ -n "${prefix}" && "${prefix}" != */ ]]; then
    prefix="${prefix}/"
  fi

  src_base="s3://${bucket}/${prefix}"
  dst_base="${DEST_ROOT%/}/${bucket}/${prefix}"

  # Ensure destination directory exists locally
  mkdir -p "${dst_base}"

  # Use sync with recursive wildcard so existing files are skipped and structure is preserved
  echo "sync \"${src_base%/}/**\" \"${dst_base}\"" >> "${TMP_RUNFILE}"
done

echo "Preview of commands (first 10):" >&2
head -n 10 "${TMP_RUNFILE}" >&2 || true

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; not executing. Full command file at: ${TMP_RUNFILE}" >&2
  cat "${TMP_RUNFILE}"
  exit 0
fi

# Execute with concurrency; adjust S5CMD_NUMWORKERS to tune parallelism
NUM_WORKERS=${S5CMD_NUMWORKERS:-64}
echo "Executing s5cmd run with ${NUM_WORKERS} workers..." >&2
s5cmd --numworkers "${NUM_WORKERS}" run "${TMP_RUNFILE}"

echo "Done."


