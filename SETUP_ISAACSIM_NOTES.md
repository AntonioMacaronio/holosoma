# IsaacSim/IsaacLab Setup Notes

Date: 2026-05-12

## What Was Set Up

IsaacSim/IsaacLab training environment at `~/.holosoma_deps/miniconda3/envs/hssim`
(conda, Python 3.11).

Host context: Ubuntu 24.04.4 (GLIBC 2.39), single NVIDIA L40S, driver 580.126.09.
This satisfies the IsaacSim/IsaacLab OS requirement (Ubuntu 22.04+) and the pip
install path's GLIBC 2.35+ requirement, so `scripts/setup_isaacsim.sh` runs here.

## Command Run

```bash
bash scripts/setup_isaacsim.sh
```

The script:

- Installs `isaacsim[all,extscache]==5.1.0` with torch 2.7.0+cu128
- Clones IsaacLab v2.3.0 to `~/.holosoma_deps/IsaacLab` (detached at 3c6e67b)
- Runs `./isaaclab.sh --install` with the upstream-bug workarounds the script
  embeds (setuptools<81 build constraint, flatdict 4.1.0 pin,
  `CMAKE_POLICY_VERSION_MINIMUM=3.5`, `OMNI_KIT_ACCEPT_EULA=1`)
- Installs Holosoma `src/holosoma[unitree,booster]` (editable)
- Force-upgrades `wandb` past the rl-games pin

## How To Activate

```bash
source scripts/source_isaacsim_setup.sh
```

## Validation

```bash
source scripts/source_isaacsim_setup.sh
python - <<'PY'
import torch, isaacsim, isaaclab, holosoma
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("isaacsim: imported")
print("isaaclab:", isaaclab.__version__)
print("holosoma: imported")
PY
```

Result on 2026-05-12:

- `torch 2.7.0+cu128`, CUDA available
- device 0: `NVIDIA L40S`
- `isaacsim` imported (pip reports version 5.1.0)
- `isaaclab 0.47.2`
- `holosoma` imported

## Gotchas Encountered

- **`/boot` full (exit code 100 on first run).** The script's
  `sudo apt install cmake build-essential` is a no-op on this host (both were
  already present), but the apt invocation triggered dpkg to finish several
  partially configured kernel packages. `/boot` was 100% full from stale initrd
  images, which caused initramfs regeneration to fail. Fix:
  `sudo rm -f /boot/initrd.img-<oldest>` → `sudo dpkg --configure -a` →
  `sudo apt autoremove --purge -y`. That freed ~500MB and unblocked the script.
- **Transient `warp-lang` wheel download stall.** The final
  `pip install -e src/holosoma[unitree,booster]` step broke partway through
  downloading `warp_lang-1.10.0` (132MB). Re-running the same pip command inside
  the activated `hssim` env recovered cleanly. The script does not normally
  create a sentinel file unless the whole block completes, so after patching up
  the install by hand I also ran `touch ~/.holosoma_deps/.env_setup_finished_hssim`
  to make re-runs a no-op.
- **Intentional wandb pin conflict.** After `pip install --upgrade 'wandb>=0.21.1'`
  pip reports a conflict against `holosoma==0.0.1`'s pin of `wandb==0.22.0`.
  This matches the setup script's own comment — the upgrade is deliberate, to
  override rl-games' constraint — and does not indicate a broken install.

## Artifacts Created

- `~/.holosoma_deps/miniconda3/envs/hssim` — conda env
- `~/.holosoma_deps/IsaacLab` — cloned IsaacLab v2.3.0 (detached HEAD)
- `~/.holosoma_deps/.env_setup_finished_hssim` — setup sentinel (created manually
  after the retries)
