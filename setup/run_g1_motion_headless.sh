#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-robotlab311}"
MOTION_FILE="${2:-}"

if [[ -z "${MOTION_FILE}" ]]; then
  printf 'Usage: %s [conda-env] /absolute/path/to/motion.motion\n' "$(basename "$0")" >&2
  exit 2
fi

MOTION_FILE="$(realpath "${MOTION_FILE}")"
[[ -f "${MOTION_FILE}" ]] || {
  printf '[ERROR] Motion file not found: %s\n' "${MOTION_FILE}" >&2
  exit 1
}

G1_MOTION_FILE="${MOTION_FILE}" "${ROOT_DIR}/verify_robotlab.sh" "${ENV_NAME}"
