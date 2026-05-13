# LAFAN → G1 / T1 Motion Data Pipeline

How to obtain LAFAN, retarget it to G1 and T1, convert to the WBT loader
format, split train/val, and (optionally) sync the result to S3.

This is the recipe behind the data already on this host:

```
src/holosoma/holosoma/data/motions/
├── g1_29dof/lafan_walkrun/                 16 clips (Phase 1 output)
├── g1_29dof/lafan_walkrun_split/{train,val}  13 / 3 clips (seed=0, val=0.2)
├── t1_23dof/lafan_walkrun/                 16 clips (this Phase 3 output)
└── t1_23dof/lafan_walkrun_split/{train,val}  13 / 3 clips (same split policy)
```

For the project context, see [`plan.md`](./plan.md). Phase 1 (G1) results
are at [`/home/ubuntu/sky_workdir/holosoma/phase1_data/phase1_lafan_walkrun_g1/phase1_results.md`](../phase1_data/phase1_lafan_walkrun_g1/phase1_results.md)
when the zip is unpacked.

---

## 1. Get LAFAN

LAFAN1 is Ubisoft's mocap dataset, distributed as 77 BVHs in a single zip.

```bash
# 138 MB
curl -L -o /tmp/lafan1.zip \
  https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip
unzip /tmp/lafan1.zip -d /tmp/lafan1
ls /tmp/lafan1/lafan1/*.bvh | wc -l   # → 77
```

The Phase 1 dataset on this host uses 16 of those clips: 12 walks
(`walk{1..4}_subject{1..5}`) and 4 runs (`run{1..2}_subject{1..5}`).
Subset selection lives in
[`phase1_data/phase1_lafan_walkrun_g1/scripts/retarget_lafan_walkrun_g1.sh`](../phase1_data/phase1_lafan_walkrun_g1/scripts/retarget_lafan_walkrun_g1.sh)
(implicitly — it just iterates over `demo_data/lafan/*.npy`, so what's in
that directory determines the subset).

## 2. Convert BVH → joint-positions `.npy`

The retargeter reads `.npy` arrays of human joint positions (22 joints × 3
xyz at 30 FPS), not raw BVHs. Conversion uses Ubisoft's own loader.

```bash
cd src/holosoma_retargeting/holosoma_retargeting/data_utils/
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .

source scripts/source_retargeting_setup.sh
python extract_global_positions.py \
  --input_dir /tmp/lafan1/lafan1 \
  --output_dir ../demo_data/lafan
```

This emits `demo_data/lafan/<clip>.npy` files. For the walk/run subset,
copy or symlink only those 16 files into `demo_data/lafan/`.

The `.npy` files for the walk/run subset are already on this host at
`phase1_data/phase1_lafan_walkrun_g1/raw/*.npy` (preserved from Phase 1),
so steps 1–2 can be skipped if you only need walk+run.

## 3. Set up the retargeting environment

One-time, ~5–15 min. Creates `~/.holosoma_deps/miniconda3/envs/hsretargeting`.

```bash
bash scripts/setup_retargeting.sh
```

Activate it in any new shell:

```bash
source scripts/source_retargeting_setup.sh
```

## 4. Retarget LAFAN → robot (per-frame IK)

The retargeter is `holosoma_retargeting`'s parallel CVXPY+CLARABEL SQP. CPU
only. Pin BLAS/OMP threads to 1 and cap workers — the default
`mp.cpu_count()` causes oversubscription and zero forward progress.

### G1

```bash
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python examples/parallel_robot_retarget.py \
  --robot g1 \
  --data-dir demo_data/lafan \
  --task-type robot_only \
  --data_format lafan \
  --save_dir demo_results_parallel/g1/robot_only/lafan_walkrun \
  --task-config.object-name ground \
  --task-config.ground-range -10 10 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --max-workers 8
```

### T1

Same command, swap `--robot g1` → `--robot t1` and the save dir.

```bash
python examples/parallel_robot_retarget.py \
  --robot t1 \
  --data-dir demo_data/lafan \
  --task-type robot_only \
  --data_format lafan \
  --save_dir demo_results_parallel/t1/robot_only/lafan_walkrun \
  --task-config.object-name ground \
  --task-config.ground-range -10 10 \
  --retargeter.foot-sticking-tolerance 0.02 \
  --max-workers 8
```

The robot models loaded under the hood are
`models/g1/g1_29dof.urdf` and `models/t1/t1_23dof.xml`. The joint name
mapping for each pair is in `config_types/data_type.py` under
`("lafan", "g1")` / `("lafan", "t1")`.

**Wall clock**: ~37 min/clip per worker on a 32-core host (≈3.3 it/s,
~7000–13000 frames per clip). With 8 parallel workers, 16 clips finishes
in ~2 hours.

**Output**: `<save_dir>/<clip>_original.npz`. Each NPZ has key `qpos` —
this is **not** the WBT format yet.

## 5. Convert qpos → WBT loader format

The WBT loader (`holosoma.managers.command.terms.wbt.MotionLoader`)
expects keys `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
`body_lin_vel_w`, `body_ang_vel_w`, `body_names`, `joint_names`, `fps`.
Conversion uses `convert_data_format_mj.py` (rolls forward kinematics in
MuJoCo at the target FPS).

That script normally requires a display, so use the headless wrapper from
Phase 1:

```bash
cd src/holosoma_retargeting/holosoma_retargeting

INPUT_DIR=demo_results_parallel/t1/robot_only/lafan_walkrun
OUT_DIR=/home/ubuntu/sky_workdir/holosoma/src/holosoma/holosoma/data/motions/t1_23dof/lafan_walkrun
CONVERTER=/home/ubuntu/sky_workdir/holosoma/phase1_data/phase1_lafan_walkrun_g1/scripts/convert_headless.py
mkdir -p "$OUT_DIR"

for src in "$INPUT_DIR"/*_original.npz; do
  base=$(basename "$src" _original.npz)
  out="$OUT_DIR/${base}.npz"
  [ -f "$out" ] && { echo "skip: $out"; continue; }
  python "$CONVERTER" \
    --input_file "$src" \
    --output_fps 50 \
    --output_name "$out" \
    --data_format lafan \
    --object_name ground \
    --once
done
```

For G1, swap the input/output paths and use
`models/g1/g1_29dof.urdf` (the converter picks it up via the `--robot`
flag inferred from the input). Phase 1's
[`convert_walkrun_to_wbt.sh`](../phase1_data/phase1_lafan_walkrun_g1/scripts/convert_walkrun_to_wbt.sh)
is the G1 reference.

`convert_headless.py` is just `convert_data_format_mj.py` with
`mujoco.viewer.launch_passive` monkey-patched to a no-op so it doesn't
require a display.

Output FPS is 50 by default (input is 30 from LAFAN). Body count is 51
for G1, less for T1; the loader validates it against the URDF body list.

## 6. Sanity check + train/val split

Reuse Phase 1's scripts (they are robot-agnostic):

```bash
PHASE1=/home/ubuntu/sky_workdir/holosoma/phase1_data/phase1_lafan_walkrun_g1/scripts
DATA_ROOT=/home/ubuntu/sky_workdir/holosoma/src/holosoma/holosoma/data/motions

# Numerical checks (NaN/Inf, ground penetration, world-body rest)
python "$PHASE1/sanity_check.py" --input_dir "$DATA_ROOT/t1_23dof/lafan_walkrun"

# Stratified split (seed=0, val_fraction=0.2 — same recipe as G1)
python "$PHASE1/split_train_val.py" \
  --input_dir "$DATA_ROOT/t1_23dof/lafan_walkrun" \
  --output_root "$DATA_ROOT/t1_23dof/lafan_walkrun_split" \
  --val_fraction 0.2 --seed 0

# Aggregate stats (per-clip + dataset-wide, JSON)
python "$PHASE1/dataset_stats.py" \
  --input_dir "$DATA_ROOT/t1_23dof/lafan_walkrun" \
  --split_root "$DATA_ROOT/t1_23dof/lafan_walkrun_split" \
  --output_json "$DATA_ROOT/t1_23dof/dataset_stats_t1.json"
```

The G1 split policy (seed=0, val_fraction=0.2) yields 13 train + 3 val.
Using the same seed on T1 produces a different split (16 → 13/3) because
clip names hash differently — that's fine, you only need the *fraction*
to match across robots, not the specific held-out clips.

The Phase 1 G1 val clips are `run2_subject1`, `walk1_subject2`,
`walk3_subject4`. If you want T1's val set to mirror G1's exactly (so
held-out clips cover the same human motions), pass an explicit
`--val_clips` list to `split_train_val.py`.

## 7. (Optional) Sync to S3

The cross-embodiment Phase 5 sweep wants the dataset on S3 so multiple
training nodes can fetch it without re-running retargeting.

Target: `s3://far-research-internal/antzhan/humanoid285/`

**Required permissions**: write access to that prefix. As of 2026-05-13
the `AIModelAccess` role attached to this host is read/write-denied
on this bucket — you'll need to run these uploads from a workstation or
EC2 instance with the right IAM policy attached.

Recommended layout under the prefix:

```
s3://far-research-internal/antzhan/humanoid285/
├── lafan_walkrun/
│   ├── g1_29dof/
│   │   ├── flat/                      <- mirror of lafan_walkrun/
│   │   ├── split/train/
│   │   ├── split/val/
│   │   └── dataset_stats_g1.json
│   └── t1_23dof/
│       ├── flat/
│       ├── split/train/
│       ├── split/val/
│       └── dataset_stats_t1.json
└── README.md                            <- copy of this file
```

Upload commands (run from a host with write permission):

```bash
DATA_ROOT=src/holosoma/holosoma/data/motions
S3=s3://far-research-internal/antzhan/humanoid285/lafan_walkrun

# G1
aws s3 sync "$DATA_ROOT/g1_29dof/lafan_walkrun"        "$S3/g1_29dof/flat/"        --region us-east-1
aws s3 sync "$DATA_ROOT/g1_29dof/lafan_walkrun_split"  "$S3/g1_29dof/split/"       --region us-east-1
aws s3 cp   "$DATA_ROOT/g1_29dof/dataset_stats_g1.json" "$S3/g1_29dof/"             --region us-east-1

# T1
aws s3 sync "$DATA_ROOT/t1_23dof/lafan_walkrun"        "$S3/t1_23dof/flat/"        --region us-east-1
aws s3 sync "$DATA_ROOT/t1_23dof/lafan_walkrun_split"  "$S3/t1_23dof/split/"       --region us-east-1
aws s3 cp   "$DATA_ROOT/t1_23dof/dataset_stats_t1.json" "$S3/t1_23dof/"             --region us-east-1
```

Loading from S3 in training: `MotionConfig.motion_file` and
`motion_dir` go through `holosoma.utils.path.resolve_data_file_path`,
which preserves `s3://...` paths as-is. Holosoma's loader uses
`smart_open` under the hood, so an S3 motion path works directly:

```bash
python src/holosoma/holosoma/train_agent.py exp:t1-29dof-wbt logger:wandb \
  --command.setup_terms.motion_command.params.motion_config.motion_dir=\
s3://far-research-internal/antzhan/humanoid285/lafan_walkrun/t1_23dof/split/train
```

(Note: if `MultiMotionLoader` doesn't yet support S3-prefix listing, you
may need to localize the directory first or pass each clip via a
comma-separated `motion_dir` list. Check
`src/holosoma/holosoma/managers/command/terms/wbt.py:248-254` — it uses
`Path(...).glob("*.npz")`, which is filesystem-only as of this writing.)

## 8. Reproducing from scratch

If you want to re-run the entire pipeline on a fresh host:

```bash
# 0. Setup
bash scripts/setup_retargeting.sh
bash scripts/setup_isaacsim.sh   # for training, not needed for data prep

# 1. Pull LAFAN, extract joint-position .npys
cd src/holosoma_retargeting/holosoma_retargeting/data_utils
curl -L -o /tmp/lafan1.zip \
  https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip
unzip /tmp/lafan1.zip -d /tmp/lafan1
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
source ../../scripts/source_retargeting_setup.sh
python extract_global_positions.py \
  --input_dir /tmp/lafan1/lafan1 --output_dir ../demo_data/lafan

# 2. (Optional) restrict to walk+run
mkdir -p ../demo_data/lafan_walkrun
mv ../demo_data/lafan/walk* ../demo_data/lafan/run* ../demo_data/lafan_walkrun/

# 3. Retarget to both robots (~2h CPU each, parallelizable)
cd ..
for ROBOT in g1 t1; do
  mkdir -p demo_results_parallel/$ROBOT/robot_only/lafan_walkrun
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python examples/parallel_robot_retarget.py \
    --robot $ROBOT --data-dir demo_data/lafan_walkrun \
    --task-type robot_only --data_format lafan \
    --save_dir demo_results_parallel/$ROBOT/robot_only/lafan_walkrun \
    --task-config.object-name ground --task-config.ground-range -10 10 \
    --retargeter.foot-sticking-tolerance 0.02 --max-workers 8
done

# 4. Convert each robot's qpos NPZs to WBT loader format
#    (loop from step 5 above, for both g1_29dof and t1_23dof)

# 5. Sanity check + split + stats (step 6 above, per robot)

# 6. Sync to S3 (step 7 above)
```

## Notes / gotchas

- **`max_workers` matters.** The default
  (`mp.cpu_count()`, ~32 here, ~112 on the source host) combined with
  CVXPY's internal threading collapses into oversubscription. `--max-workers 8`
  with `OMP_NUM_THREADS=1` is the safe combo.
- **`convert_data_format_mj.py` requires a display** unless you wrap it
  with the Phase 1 `convert_headless.py` shim, which monkey-patches
  `mujoco.viewer.launch_passive` to a no-op.
- **Output suffix.** Robot-only retargeting saves `<clip>_original.npz`,
  not `<clip>.npz`. Strip `_original` when naming WBT outputs.
- **Tyro flag style.** Use `--retargeter.no-visualize` (group-prefix-then-no),
  not `--no-retargeter.visualize`.
- **5 walk clips have crouching segments** where the actor squats and
  pelvis drops below 0.4 m. These are real LAFAN behaviors, not
  retargeting bugs. See Phase 1 results for the exact frame counts. For a
  pure locomotion policy you may want to trim them; for general WBT,
  they're useful variety.
