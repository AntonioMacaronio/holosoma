# Local Setup Notes

Date: 2026-05-12

## What Was Set Up

This repository was set up with uv-managed, repo-local virtual environments:

- Inference/deployment: `.venv/hsinference`
- MuJoCo/core Holosoma: `.venv/hsmujoco`
- IsaacGym training: `~/.holosoma_deps/miniconda3/envs/hsgym`

The recommended local training workflow for this host is IsaacGym. It is
GPU-accelerated, imports successfully on the H200 GPUs in this machine, and it
matches the existing locomotion experiment presets in the repository.

The MuJoCo/core environment is the CPU MuJoCo ClassicBackend setup. It was
created with `--no-warp`, so the GPU-accelerated MuJoCo Warp backend was not
installed. Some CUDA/PyTorch and `warp-lang` packages are still present because
they are regular Holosoma dependencies, but that does not mean this environment
is running MuJoCo simulation through MuJoCo Warp.

## Default Workflow

IsaacGym is the recommended training workflow for this host. The bare training
config and named locomotion presets already point at IsaacGym. The direct
simulation runner still defaults to MuJoCo for sim2sim-style use; pass
`simulator:isaacgym` when you want direct simulation through IsaacGym.

This host is Ubuntu 20.04.6, while the repo README says IsaacSim setup requires
Ubuntu 22.04 or later, so I did not run `scripts/setup_isaacsim.sh` here.

IsaacLab is part of that same risk. The repo's IsaacSim setup script installs
`isaacsim[all,extscache]==5.1.0`, clones IsaacLab `v2.3.0`, and creates a
Python 3.11 conda environment. Current IsaacLab v2.3.0 docs list Ubuntu 22.04
Linux x64 or Windows 11 as the basic OS requirement, and the Isaac Sim pip
install path requires GLIBC 2.35+. Ubuntu 20.04 normally ships GLIBC 2.31, so a
local pip-based IsaacSim/IsaacLab install on this host is expected to be fragile.
Use an Ubuntu 22.04/24.04 host or a supported IsaacSim/IsaacLab container for
the IsaacSim training workflow.

I also installed a uv-managed Python 3.12.11 interpreter because the system Python
3.12 did not include development headers needed to build `evdev`.

## Commands Run

```bash
uv python install 3.12
bash scripts/setup_inference_via_uv.sh --python 3.12 --reinstall
uv pip install --python .venv/hsinference/bin/python -e 'src/holosoma_inference[dev]'

bash scripts/setup_mujoco_via_uv.sh --python 3.12 --no-warp --no-robot-sdks

bash scripts/setup_isaacgym.sh
```

## How To Activate

Inference:

```bash
source scripts/source_inference_uv_setup.sh
```

MuJoCo/core Holosoma:

```bash
source scripts/source_mujoco_uv_setup.sh
```

IsaacGym:

```bash
source scripts/source_isaacgym_setup.sh
```

## Validation

Inference environment:

```bash
cd src/holosoma_inference
source ../../scripts/source_inference_uv_setup.sh
python -m pytest -q --confcutdir=. \
  holosoma_inference/config/tests \
  holosoma_inference/inputs/tests \
  holosoma_inference/policies/tests \
  holosoma_inference/utils/tests
```

Result: `201 passed, 1 warning`

MuJoCo/core environment:

```bash
source scripts/source_mujoco_uv_setup.sh
python -m pytest -q \
  src/holosoma/holosoma/utils/tests/test_rotations.py \
  src/holosoma/holosoma/utils/tests/test_torch_utils.py \
  src/holosoma/tests/utils/test_file_cache.py
```

Result: `51 passed, 49 warnings`

Import checks also passed for:

- `holosoma_inference`
- `pinocchio 3.9.0`
- `holosoma`
- `mujoco 3.8.1`
- `torch 2.10.0+cu128`

IsaacGym environment:

```bash
source scripts/source_isaacgym_setup.sh
python - <<'PY'
import isaacgym
print("isaacgym imported")
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
PY
```

Result:

- `isaacgym` imported successfully
- `torch 2.4.1+cu121`
- CUDA available: `True`
- CUDA device count: `8`
- CUDA device 0: `NVIDIA H200`

IsaacGym GPU simulation smoke test:

```bash
source scripts/source_isaacgym_setup.sh
python -m pytest -q src/holosoma/holosoma/envs/tests/test_e2e.py::test_e2e_step
```

Result: `1 passed, 3 warnings in 87.20s`

## Conda Impact

The IsaacGym setup script did create a repo-managed Miniconda installation and
environment under `~/.holosoma_deps`:

- `~/.holosoma_deps/miniconda3`
- `~/.holosoma_deps/miniconda3/envs/hsgym`
- `~/.holosoma_deps/isaacgym`

This is isolated from any pre-existing system/user conda installation.

The setup did create uv/local artifacts:

- `.venv/hsinference`
- `.venv/hsmujoco`
- a uv-managed Python under `~/.local/share/uv/python`
- normal uv package cache entries

The repo's `.gitignore` already ignores `.venv/`.

## Not Configured

- IsaacSim setup
- MuJoCo Warp GPU backend
- Robot SDKs in the MuJoCo/core environment
- Retargeting environment
- Pre-commit hooks

Those can be added later if needed for the specific workflow.

## MuJoCo Warp Note

Holosoma has MJWarp support, but the repository documentation labels it beta.
For this initial setup I avoided installing MuJoCo Warp because the CPU
ClassicBackend path is lower risk for import checks, inference-style MuJoCo
simulation, and local smoke tests. For GPU-accelerated training on this host,
IsaacGym is the recommended path.
