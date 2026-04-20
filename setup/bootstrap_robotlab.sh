#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-robotlab311}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
ENV_WAS_CREATED="0"

TARGET_PYTHON="3.11"
TARGET_ISAACSIM="5.1.0.0"
TARGET_TORCH="2.7.0"
TARGET_TORCHVISION="0.22.0"
TARGET_TORCHAUDIO="2.7.0"
TARGET_TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TARGET_CLICK="8.1.7"
TARGET_PACKAGING="23.0"
TARGET_RTREE="1.3.0"
TARGET_SENTRY="2.29.1"
TARGET_WHEEL_SPEC="wheel<0.46"

PROTOMOTIONS_REPO="${PROTOMOTIONS_REPO:-/home/amunyoz/ProtoMotions}"
ISAACLAB_REPO="${ISAACLAB_REPO:-/home/amunyoz/IsaacLab}"
WSL_LIB_DIR="/usr/lib/wsl/lib"
RUNTIME_CACHE_ROOT="/tmp/robotlab-cache/${ENV_NAME}"

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

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

ensure_repo_layout() {
  [[ -f "${PROTOMOTIONS_REPO}/setup.py" ]] || die "ProtoMotions repo not found at ${PROTOMOTIONS_REPO}"
  [[ -f "${PROTOMOTIONS_REPO}/requirements_isaaclab.txt" ]] || die "Missing ${PROTOMOTIONS_REPO}/requirements_isaaclab.txt"
  [[ -f "${PROTOMOTIONS_REPO}/protomotions/inference_agent.py" ]] || die "Missing ${PROTOMOTIONS_REPO}/protomotions/inference_agent.py"
  [[ -f "${ISAACLAB_REPO}/environment.yml" ]] || die "IsaacLab repo not found at ${ISAACLAB_REPO}"
  [[ -f "${ISAACLAB_REPO}/scripts/tutorials/00_sim/create_empty.py" ]] || die "Missing Isaac Lab create_empty.py entrypoint"
  [[ -f "${ISAACLAB_REPO}/source/isaaclab/setup.py" ]] || die "Missing Isaac Lab editable package source/isaaclab"
}

ensure_wsl_gpu_prereqs() {
  [[ -d "${WSL_LIB_DIR}" ]] || die "Missing ${WSL_LIB_DIR}; this setup expects WSL GPU libraries."
  export LD_LIBRARY_PATH="${WSL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  export PATH="${WSL_LIB_DIR}:${PATH}"
  if ! grep -qi microsoft /proc/version 2>/dev/null; then
    warn "WSL was not detected from /proc/version. Continuing, but this script is tuned for WSL Ubuntu."
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi is not available even after adding ${WSL_LIB_DIR} to PATH."
  fi
  local smi_out
  if ! smi_out="$(nvidia-smi 2>&1)"; then
    printf '%s\n' "${smi_out}" >&2
    die "nvidia-smi failed. Fix the Windows/WSL NVIDIA driver bridge before installing Isaac Sim."
  fi
  log "nvidia-smi succeeded."
}

load_conda() {
  require_cmd conda
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

inspect_existing_env() {
  conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys

def get(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

print(f"python={sys.version.split()[0]}")
for pkg in ("isaacsim", "torch", "isaaclab", "protomotions"):
    print(f"{pkg}={get(pkg)}")
PY
}

enforce_existing_env_compatibility() {
  local summary
  summary="$(inspect_existing_env)"
  printf '%s\n' "${summary}"

  local python_v isaacsim_v
  python_v="$(printf '%s\n' "${summary}" | awk -F= '$1=="python"{print $2}')"
  isaacsim_v="$(printf '%s\n' "${summary}" | awk -F= '$1=="isaacsim"{print $2}')"

  [[ "${python_v}" == ${TARGET_PYTHON}.* ]] || die "Existing env ${ENV_NAME} uses Python ${python_v}; required is ${TARGET_PYTHON}.x."

  if [[ -n "${isaacsim_v}" && "${isaacsim_v}" == 4.5.* ]]; then
    die "Existing env ${ENV_NAME} has Isaac Sim ${isaacsim_v}. That is incompatible with the shared Python ${TARGET_PYTHON} stack because ${ISAACLAB_REPO}/isaaclab.sh forces Python 3.10 for Isaac Sim 4.5."
  fi

  if [[ -n "${isaacsim_v}" && "${isaacsim_v}" != "${TARGET_ISAACSIM}" ]]; then
    die "Existing env ${ENV_NAME} has Isaac Sim ${isaacsim_v}. This setup pins ${TARGET_ISAACSIM} because the local IsaacLab repo documents 4.5/5.0/5.1 compatibility and the known-good shared env here is ${TARGET_ISAACSIM}."
  fi
}

create_or_validate_env() {
  if env_exists; then
    if [[ "${FORCE_RECREATE}" == "1" ]]; then
      log "Removing existing conda env ${ENV_NAME}"
      conda env remove -y -n "${ENV_NAME}"
      ENV_WAS_CREATED="1"
    else
      log "Conda env ${ENV_NAME} already exists; validating compatibility."
      enforce_existing_env_compatibility
      return
    fi
  fi

  log "Creating conda env ${ENV_NAME} from ${ROOT_DIR}/environment.yml"
  conda env create -y -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.yml"
  ENV_WAS_CREATED="1"
}

install_wsl_activation_hooks() {
  local conda_prefix
  conda_prefix="$(conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import os
print(os.environ["CONDA_PREFIX"])
PY
)"

  mkdir -p "${conda_prefix}/etc/conda/activate.d" "${conda_prefix}/etc/conda/deactivate.d" "${conda_prefix}/.cache/matplotlib"

  cat > "${conda_prefix}/etc/conda/activate.d/robotlab_wsl.sh" <<EOF
#!/usr/bin/env bash
export LD_LIBRARY_PATH="${WSL_LIB_DIR}:\${LD_LIBRARY_PATH:-}"
export PATH="${WSL_LIB_DIR}:\${PATH}"
export ROBOTLAB_CACHE_ROOT="${RUNTIME_CACHE_ROOT}"
export MPLCONFIGDIR="\${ROBOTLAB_CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="\${ROBOTLAB_CACHE_ROOT}/xdg"
export OMNI_KIT_ACCEPT_EULA=YES
export PROTOMOTIONS_DISABLE_TORCH_COMPILE=1
export ISAACLAB_PATH="${ISAACLAB_REPO}"
export PROTOMOTIONS_PATH="${PROTOMOTIONS_REPO}"
mkdir -p "\${MPLCONFIGDIR}" "\${XDG_CACHE_HOME}"
EOF

  cat > "${conda_prefix}/etc/conda/deactivate.d/robotlab_wsl.sh" <<'EOF'
#!/usr/bin/env bash
unset OMNI_KIT_ACCEPT_EULA
unset ISAACLAB_PATH
unset PROTOMOTIONS_PATH
EOF
}

install_packages() {
  log "Upgrading base packaging tools"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install --upgrade "pip<26" "setuptools==69.5.1" "${TARGET_WHEEL_SPEC}"

  log "Installing PyTorch ${TARGET_TORCH} CUDA 12.8 wheels"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install \
    --index-url "${TARGET_TORCH_INDEX}" \
    "torch==${TARGET_TORCH}" \
    "torchvision==${TARGET_TORCHVISION}" \
    "torchaudio==${TARGET_TORCHAUDIO}"

  log "Installing Isaac Sim ${TARGET_ISAACSIM}"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install \
    "isaacsim==${TARGET_ISAACSIM}" \
    "isaacsim-rl==${TARGET_ISAACSIM}"

  log "Installing local editable Isaac Lab base package"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -e "${ISAACLAB_REPO}/source/isaaclab"

  log "Installing ProtoMotions Isaac Lab runtime dependencies with conflict-safe filtering"
  local filtered_requirements
  filtered_requirements="$(mktemp)"
  awk '
    BEGIN { IGNORECASE = 1 }
    /^[[:space:]]*#/ { print; next }
    /^[[:space:]]*$/ { print; next }
    $0 ~ /^wheel([[:space:]]|=|<|>|!|$)/ { next }
    $0 ~ /^rtree([[:space:]]|=|<|>|!|$)/ { next }
    $0 ~ /^sentry-sdk([[:space:]]|=|<|>|!|$)/ { next }
    $0 ~ /^typer([[:space:]]|=|<|>|!|$)/ { next }
    { print }
  ' "${PROTOMOTIONS_REPO}/requirements_isaaclab.txt" > "${filtered_requirements}"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -r "${filtered_requirements}"
  rm -f "${filtered_requirements}"

  log "Re-applying conflict-safe runtime pins required by Isaac Sim 5.1"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install \
    "click==${TARGET_CLICK}" \
    "packaging==${TARGET_PACKAGING}" \
    "rtree==${TARGET_RTREE}" \
    "sentry-sdk==${TARGET_SENTRY}" \
    "${TARGET_WHEEL_SPEC}"

  log "Removing typer because its newer click requirement conflicts with Isaac Sim and it is not needed for runtime verification"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip uninstall -y typer || true

  log "Installing local editable ProtoMotions package"
  conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -e "${PROTOMOTIONS_REPO}"
}

audit_dependency_conflicts() {
  log "Auditing package metadata conflicts"
  local pip_check_output rc
  local pip_check_log
  pip_check_log="$(mktemp)"

  if conda run --no-capture-output -n "${ENV_NAME}" python -m pip check >"${pip_check_log}" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  pip_check_output="$(cat "${pip_check_log}")"
  rm -f "${pip_check_log}"

  if [[ "${rc}" -eq 0 ]]; then
    printf '%s\n' "${pip_check_output}"
    return
  fi

  printf '%s\n' "${pip_check_output}"

  local filtered
  filtered="$(printf '%s\n' "${pip_check_output}" | sed '/^ERROR conda\.cli\.main_run:/d')"
  if printf '%s\n' "${filtered}" | grep -Fq 'fastapi 0.115.7 has requirement starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1.'; then
    local residual
    residual="$(
      {
        printf '%s\n' "${filtered}" \
          | grep -Fv 'fastapi 0.115.7 has requirement starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1.' \
          || true
      } | sed '/^[[:space:]]*$/d'
    )"
    if [[ -z "${residual}" ]]; then
      warn "Ignoring the known metadata conflict between Isaac Sim's fastapi pin and Isaac Lab's starlette pin; runtime verification is used as the source of truth."
      return
    fi
  fi

  die "Unexpected package metadata conflicts remain after installation. Resolve them before using this environment."
}

verify_versions() {
  log "Checking installed package versions"
  conda run --no-capture-output -n "${ENV_NAME}" python - <<PY
from importlib.metadata import version
import sys

target_python = "${TARGET_PYTHON}"
target_isaacsim = "${TARGET_ISAACSIM}"
target_torch = "${TARGET_TORCH}"

python_v = sys.version.split()[0]
if not python_v.startswith(target_python + "."):
    raise SystemExit(f"Python mismatch: expected {target_python}.x, got {python_v}")

isaacsim_v = version("isaacsim")
if isaacsim_v.startswith("4.5."):
    raise SystemExit(
        "Isaac Sim 4.5 detected. The local Isaac Lab launcher forces Python 3.10 for 4.5, "
        "which conflicts with the required Python 3.11 shared environment."
    )
if isaacsim_v != target_isaacsim:
    raise SystemExit(
        f"Isaac Sim mismatch: expected {target_isaacsim}, got {isaacsim_v}. "
        "This setup intentionally refuses newer major versions such as 6.x."
    )

torch_v = version("torch")
if not torch_v.startswith(target_torch):
    raise SystemExit(f"Torch mismatch: expected {target_torch}.x, got {torch_v}")

for pkg in ("isaaclab", "protomotions"):
    print(f"{pkg}={version(pkg)}")
print(f"python={python_v}")
print(f"isaacsim={isaacsim_v}")
print(f"torch={torch_v}")
PY
}

run_post_install_verification() {
  log "Running verification script"
  "${ROOT_DIR}/verify_robotlab.sh" "${ENV_NAME}"
}

main() {
  require_cmd git
  require_cmd awk
  require_cmd timeout
  ensure_repo_layout
  ensure_wsl_gpu_prereqs
  load_conda
  create_or_validate_env
  install_wsl_activation_hooks
  if [[ "${ENV_WAS_CREATED}" == "1" || "${FORCE_REINSTALL}" == "1" ]]; then
    install_packages
  else
    log "Skipping package reinstalls because ${ENV_NAME} already matches the pinned stack. Set FORCE_REINSTALL=1 to reinstall packages."
  fi
  verify_versions
  audit_dependency_conflicts
  run_post_install_verification
  log "Bootstrap finished successfully for conda env ${ENV_NAME}"
}

main "$@"
