#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-robotlab311}"

PROTOMOTIONS_REPO="${PROTOMOTIONS_REPO:-/home/amunyoz/ProtoMotions}"
ISAACLAB_REPO="${ISAACLAB_REPO:-/home/amunyoz/IsaacLab}"
WSL_LIB_DIR="/usr/lib/wsl/lib"
RUNTIME_CACHE_ROOT="${ROBOTLAB_CACHE_ROOT:-/tmp/robotlab-cache/${ENV_NAME}}"

CREATE_EMPTY="${ISAACLAB_REPO}/scripts/tutorials/00_sim/create_empty.py"
INFERENCE_AGENT="${PROTOMOTIONS_REPO}/protomotions/inference_agent.py"
G1_CHECKPOINT="${PROTOMOTIONS_REPO}/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt"
G1_MOTION_FILE="${PROTOMOTIONS_REPO}/data/motion_for_trackers/g1_bones_seed_mini.pt"

log() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

load_conda() {
  command -v conda >/dev/null 2>&1 || die "conda is required"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

setup_runtime_env() {
  [[ -d "${WSL_LIB_DIR}" ]] || die "Missing ${WSL_LIB_DIR}"
  export LD_LIBRARY_PATH="${WSL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  export PATH="${WSL_LIB_DIR}:${PATH}"
  export ROBOTLAB_CACHE_ROOT="${RUNTIME_CACHE_ROOT}"
  export MPLCONFIGDIR="${ROBOTLAB_CACHE_ROOT}/matplotlib"
  export XDG_CACHE_HOME="${ROBOTLAB_CACHE_ROOT}/xdg"
  export OMNI_KIT_ACCEPT_EULA=YES
  export PROTOMOTIONS_DISABLE_TORCH_COMPILE=1
  mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
}

check_repo_layout() {
  [[ -f "${CREATE_EMPTY}" ]] || die "Isaac Lab verification target missing: ${CREATE_EMPTY}"
  [[ -f "${INFERENCE_AGENT}" ]] || die "ProtoMotions verification target missing: ${INFERENCE_AGENT}"
}

check_nvidia_smi() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found after exporting ${WSL_LIB_DIR} into PATH"
  local smi_out
  if ! smi_out="$(nvidia-smi 2>&1)"; then
    printf '%s\n' "${smi_out}" >&2
    die "nvidia-smi failed. On this host the common failure is WSL GPU bridge/driver not loaded."
  fi
  log "nvidia-smi succeeded."
}

check_python_and_versions() {
  python - <<'PY'
from importlib.metadata import version
import sys

python_v = sys.version.split()[0]
if not python_v.startswith("3.11."):
    raise SystemExit(f"Python mismatch: expected 3.11.x, got {python_v}")

isaacsim_v = version("isaacsim")
if isaacsim_v.startswith("4.5."):
    raise SystemExit(
        "Isaac Sim 4.5 is incompatible with this shared stack because the local Isaac Lab launcher forces Python 3.10 for 4.5."
    )
if isaacsim_v != "5.1.0.0":
    raise SystemExit(
        f"Isaac Sim mismatch: expected 5.1.0.0, got {isaacsim_v}. This setup does not claim compatibility for 6.x."
    )

print(f"python={python_v}")
print(f"isaacsim={isaacsim_v}")
print(f"torch={version('torch')}")
print(f"isaaclab={version('isaaclab')}")
print(f"protomotions={version('protomotions')}")
PY
}

check_torch_cuda() {
  local torch_out
  torch_out="$(python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_name={torch.cuda.get_device_name(0)}")
PY
)"
  printf '%s\n' "${torch_out}"
  printf '%s\n' "${torch_out}" | grep -Fq 'cuda_available=True' || die "torch.cuda.is_available() is False"
}

run_with_capture() {
  local name="$1"
  local logfile="$2"
  local timeout_s="$3"
  shift 3
  log "Running ${name}"

  set +e
  setsid "$@" >"${logfile}" 2>&1 &
  local pid=$!
  local rc=0
  local elapsed=0

  while kill -0 "${pid}" 2>/dev/null; do
    if (( elapsed >= timeout_s )); then
      kill -TERM -- "-${pid}" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      rc=124
      break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if (( rc == 0 )); then
    wait "${pid}"
    rc=$?
  fi
  set -e

  cat "${logfile}"
  return "${rc}"
}

classify_runtime_failure() {
  local logfile="$1"
  if grep -Eq 'Found no NVIDIA driver|No supported gpu backend found|No CUDA devices found|cudaErrorInsufficientDriver|CUDA driver version is insufficient|GPU access blocked by the operating system' "${logfile}"; then
    die "GPU runtime failure detected while running $(basename "${logfile}"). The package stack is installed, but the WSL NVIDIA driver bridge is not usable."
  fi
  if grep -Eq 'Traceback|FileNotFoundError|ModuleNotFoundError|ImportError|ResolutionImpossible|UnsatisfiableError|PackageNotFoundError' "${logfile}"; then
    die "Dependency failure detected while running $(basename "${logfile}"). See the log above for the missing or conflicting package."
  fi
}

verify_isaaclab_headless() {
  local logfile
  logfile="$(mktemp)"
  if run_with_capture \
    "Isaac Lab create_empty.py --headless" \
    "${logfile}" \
    45 \
    env PYTHONUNBUFFERED=1 python "${CREATE_EMPTY}" --headless; then
    :
  else
    local rc=$?
    classify_runtime_failure "${logfile}"
    if [[ "${rc}" -eq 124 ]] && grep -Fq '[INFO]: Setup complete...' "${logfile}"; then
      log "Isaac Lab headless startup reached setup and then timed out as expected."
      rm -f "${logfile}"
      return
    fi
    rm -f "${logfile}"
    die "Isaac Lab headless verification failed."
  fi
  grep -Fq '[INFO]: Setup complete...' "${logfile}" || die "Isaac Lab did not print setup confirmation."
  rm -f "${logfile}"
}

verify_protomotions_g1() {
  [[ -f "${G1_CHECKPOINT}" ]] || { warn "Skipping ProtoMotions G1 inference: checkpoint not found at ${G1_CHECKPOINT}"; return; }
  [[ -f "${G1_MOTION_FILE}" ]] || { warn "Skipping ProtoMotions G1 inference: motion file not found at ${G1_MOTION_FILE}"; return; }

  local logfile
  logfile="$(mktemp)"
  if run_with_capture \
    "ProtoMotions G1 inference in Isaac Lab headless mode" \
    "${logfile}" \
    60 \
    bash -lc "cd '${PROTOMOTIONS_REPO}' && exec env PYTHONUNBUFFERED=1 python '${INFERENCE_AGENT}' \
      --checkpoint "${G1_CHECKPOINT}" \
      --motion-file "${G1_MOTION_FILE}" \
      --simulator isaaclab \
      --num-envs 1 \
      --headless"; then
    :
  else
    local rc=$?
    classify_runtime_failure "${logfile}"
    if [[ "${rc}" -eq 124 ]] && ! grep -Fq 'Traceback' "${logfile}"; then
      log "ProtoMotions G1 inference launched and kept running until timeout."
      rm -f "${logfile}"
      return
    fi
    rm -f "${logfile}"
    die "ProtoMotions G1 inference verification failed."
  fi
  rm -f "${logfile}"
}

main() {
  load_conda
  setup_runtime_env
  check_repo_layout
  check_nvidia_smi
  check_python_and_versions
  check_torch_cuda
  verify_isaaclab_headless
  verify_protomotions_g1
  log "Verification completed successfully for ${ENV_NAME}"
}

main "$@"
