# RobotLab WSL Bootstrap

This directory contains a conservative WSL Ubuntu setup for a shared robotics simulation environment covering:

- ProtoMotions from the local repo at `/home/amunyoz/ProtoMotions`
- Isaac Lab from the local repo at `/home/amunyoz/IsaacLab`
- Isaac Sim via pip
- G1 inference through `protomotions/inference_agent.py` when the local checkpoint and motion file are present

## Compatibility Decision

The setup is intentionally pinned to:

- Python `3.11`
- Isaac Sim `5.1.0.0`
- Torch `2.7.0+cu128`
- Isaac Lab local repo `main` at commit `4df6560e187f2cc66685b41b21b259f4485d0c22`
- ProtoMotions local repo `main` at commit `b59d44f5b3be5a9a33a3fac6a82d40e5aecaba63`

Why this pin:

- The local Isaac Lab repo advertises Python `3.11` and Isaac Sim `5.1.0` in its top-level README.
- The local Isaac Lab `environment.yml` also pins Python `3.11`.
- The local Isaac Lab launcher script `isaaclab.sh` contains an explicit compatibility branch that forces Python `3.10` when Isaac Sim `4.5` is detected.
- ProtoMotions does not require a newer Python than `3.11`; its local `setup.py` only declares `python_requires=">=3.8"`.
- A pre-existing local conda env already proves that this exact shared stack can coexist:
  - Python `3.11.15`
  - Isaac Sim `5.1.0.0`
  - Isaac Lab `0.54.3`
  - ProtoMotions `3.1`
  - Torch `2.7.0+cu128`

## Exact Incompatibilities the Scripts Reject

- Isaac Sim `4.5.x` is rejected because the local `IsaacLab/isaaclab.sh` forces Python `3.10` for that branch. That conflicts with the required Python `3.11` shared stack.
- Isaac Sim `6.x` is rejected because the local Isaac Lab repo metadata and README only document compatibility through `5.1`, and this setup is intentionally choosing compatibility over latest.
- Any environment where `torch.cuda.is_available()` is false is rejected at verification time, because Isaac Sim and the Isaac Lab runtime need usable GPU access in WSL.

## ProtoMotions on RTX 5090

On this host, ProtoMotions' `torch.compile` path trips a Triton compiler failure on compute capability `120` (RTX 5090 / Blackwell) under the pinned Torch `2.7.0+cu128` stack. To keep the verified Isaac Sim `5.1.0.0` + Python `3.11` environment, the bootstrap exports:

- `PROTOMOTIONS_DISABLE_TORCH_COMPILE=1`

That forces ProtoMotions' component manager to use eager mode instead of `torch.compile`. This avoids inventing a new, unverified Torch/Isaac Sim combination.

## Files

- [bootstrap_robotlab.sh](/home/amunyoz/robot-setup/bootstrap_robotlab.sh)
- [verify_robotlab.sh](/home/amunyoz/robot-setup/verify_robotlab.sh)
- [environment.yml](/home/amunyoz/robot-setup/environment.yml)

## What Bootstrap Does

`bootstrap_robotlab.sh`:

1. Validates the local ProtoMotions and Isaac Lab repo layouts.
2. Verifies WSL GPU plumbing:
   - checks `/usr/lib/wsl/lib`
   - prepends it to `LD_LIBRARY_PATH`
   - prepends it to `PATH`
   - runs `nvidia-smi`
3. Creates or validates a conda env from `environment.yml`.
4. Installs packages in a fixed order:
   - PyTorch `2.7.0` / CUDA 12.8 wheels
   - Isaac Sim `5.1.0.0`
   - local editable Isaac Lab base package from `source/isaaclab`
   - ProtoMotions Isaac Lab requirements
   - local editable ProtoMotions package
5. Writes conda activation hooks so future activations keep the WSL library path.
6. Runs `verify_robotlab.sh`.

## What Verification Does

`verify_robotlab.sh` checks:

- `nvidia-smi`
- Python version
- Isaac Sim version
- `torch.cuda.is_available()`
- Isaac Lab headless startup via `scripts/tutorials/00_sim/create_empty.py`
- ProtoMotions G1 inference via `protomotions/inference_agent.py` when the local G1 assets exist

Both runtime checks are executed with timeouts because the entrypoints are long-running programs. A timeout after successful startup is treated as success.

## Usage

```bash
chmod +x bootstrap_robotlab.sh verify_robotlab.sh
./bootstrap_robotlab.sh
```

Optional:

- Use another env name: `./bootstrap_robotlab.sh my_robotlab_env`
- Recreate an existing env: `FORCE_RECREATE=1 ./bootstrap_robotlab.sh`
- Reinstall packages into an already compatible env: `FORCE_REINSTALL=1 ./bootstrap_robotlab.sh`
- Override repo locations:
  - `PROTOMOTIONS_REPO=/path/to/ProtoMotions`
  - `ISAACLAB_REPO=/path/to/IsaacLab`

On reruns, the bootstrap now validates the existing env and skips package reinstalls by default if the pinned Python and Isaac Sim versions already match. This avoids re-triggering the known pip metadata conflicts between Isaac Sim, Isaac Lab, and ProtoMotions dependency pins.

When `FORCE_REINSTALL=1` is used, the bootstrap now installs a conflict-safe subset of `ProtoMotions/requirements_isaaclab.txt` and preserves Isaac Sim 5.1's required pins for:

- `click==8.1.7`
- `packaging==23.0`
- `rtree==1.3.0`
- `sentry-sdk==2.29.1`
- `wheel<0.46`

It also removes `typer` during the forced reinstall path because newer `typer` releases require newer `click` than Isaac Sim 5.1 allows, and `typer` is not needed for runtime verification.

After installation, the bootstrap runs `pip check` and allows only one known irreducible metadata conflict:

- Isaac Sim 5.1 pulls `fastapi==0.115.7`, which requires `starlette<0.46.0`
- local Isaac Lab `0.54.3` requires `starlette==0.49.1`

That pair is currently unsatisfiable at the metadata level, so the script treats runtime verification as the source of truth for the final environment.

## Local Inspection Summary

ProtoMotions local structure includes:

- `setup.py`
- `requirements_isaaclab.txt`
- `protomotions/inference_agent.py`
- G1 checkpoint at `/home/amunyoz/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt`
- G1 motion file at `/home/amunyoz/ProtoMotions/data/motion_for_trackers/g1_bones_seed_mini.pt`

Isaac Lab local structure includes:

- top-level `environment.yml`
- editable package source at `source/isaaclab`
- headless smoke test at `scripts/tutorials/00_sim/create_empty.py`

The bootstrap uses editable installs where they are appropriate:

- `pip install -e /home/amunyoz/IsaacLab/source/isaaclab`
- `pip install -e /home/amunyoz/ProtoMotions`

## Current Host Status

On this WSL host, the package compatibility is fine, but GPU access is currently broken:

- `/usr/lib/wsl/lib` exists
- `nvidia-smi` currently fails with `GPU access blocked by the operating system`
- `torch.cuda.is_available()` is currently `False`
- Isaac Lab headless startup fails with `Found no NVIDIA driver on your system`
- ProtoMotions G1 inference fails with `RuntimeError: No supported gpu backend found!`

That is a host driver/runtime problem, not an unresolved Python dependency conflict. The scripts are written to stop on that condition with a specific failure reason.
