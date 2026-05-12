# Cross-Embodiment Policy Transfer: G1 → T1

Working breakdown of the research pipeline. Phases are roughly sequential, but
items within a phase can often be parallelized. Open decisions are flagged
inline — resolve them before the phase that depends on them.

---

## Phase 0 — Decisions to lock in before coding

These change the rest of the design, so pin them down first.

- **0.1 Task framing.** Is the G1 policy:
  - (a) a whole-body motion-tracking policy (WBT) conditioned on future
    reference frames, or
  - (b) a root-velocity-conditioned locomotion policy that uses the motion
    data as a style/imitation prior (AMP / DeepMimic-style)?
  - The existing infra in `src/holosoma/holosoma/config_values/wbt/g1/` is
    (a). (b) would need a new task setup.
- **0.2 What transfers across embodiments.** G1 and T1 both have 29 DOF but
  different joint semantics (G1 has waist pitch/roll, T1 has only waist yaw;
  different foot/knee link names). "Reinit the last few layers" isn't quite
  right — the *input* proprioception dims are also semantically different.
  Decide on one:
  - reinit first + last layers, keep the middle as transferable backbone
  - use an embodiment-agnostic observation representation (e.g., keypoints
    in root frame) so only the action head needs reinit
  - train with explicit embodiment embedding
- **0.3 Baselines needed for the "10% data" claim.** Minimum set:
  `{T1-scratch @ 100%, T1-scratch @ 10%, G1→T1-finetune @ 10%}`. Better:
  sweep `{1, 5, 10, 25, 50, 100}%` and plot the learning curve.
- **0.4 Success metrics.** Pick upfront, e.g. tracking MPJPE, root-pos drift,
  survival time, reward at convergence, sample efficiency (steps to reach
  threshold). Without these pinned, the ablation story is fuzzy.
- **0.5 FPO scope.** Is flow-matching RL a required result or a stretch?
  Recommend: stretch. See Phase 5.

---

## Phase 1 — G1 motion dataset from LAFAN

Goal: produce a directory of NPZ files in the holosoma WBT format, each a
retargeted G1 motion from LAFAN.

- **1.1** Inventory LAFAN clips — pick the subset you want (walking, running,
  dance, etc.). Note which motions are feasible on G1 (no climbing, etc.).
- **1.2** Run existing LAFAN→G1 retargeting via
  `src/holosoma_retargeting/` (the `("lafan", "g1")` mapping in
  `config_types/data_type.py` is already wired). Confirm output format
  matches what the WBT command loader expects at
  `src/holosoma/holosoma/managers/command/terms/wbt.py` (keys: `fps`,
  `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`,
  `body_ang_vel_w`, `body_names`, `joint_names`).
- **1.3** Build a batch conversion script (LAFAN clip list → NPZ directory).
  Decide train/val split at the clip level, not the frame level.
- **1.4** Visual sanity-check: replay a handful of retargeted clips in
  MuJoCo (`holosoma/run_sim.py` or a small playback utility) and eyeball
  for foot-skate, penetration, unreachable poses. Discard or trim bad clips.
- **1.5** Compute and log dataset stats: total minutes, motion-type mix,
  velocity distribution. You'll want this for the paper.

---

## Phase 2 — G1 motion-tracking policy

Goal: a trained G1 policy + ONNX export that tracks LAFAN-retargeted motion.

- **2.1** Read and understand the existing G1 WBT config stack:
  `config_values/wbt/g1/{experiment,reward,observation,command,termination,randomization,curriculum}.py`.
  Note what's hardcoded vs. parameterized.
- **2.2** Wire the Phase 1 dataset into the command manager. Confirm clip
  sampling, looping, and reset behavior.
- **2.3** Short smoke-train on a single clip to verify the pipeline
  end-to-end before committing GPU hours.
- **2.4** Full PPO training run on the G1 dataset using `train_agent.py`.
  Log to W&B. Save checkpoints.
- **2.5** Evaluate with `eval_agent.py` on held-out clips. Record the
  Phase 0.4 metrics — these are your "upper-bound" reference numbers.
- **2.6** Export to ONNX (existing infra in `holosoma_inference/`).
  Sanity-check it matches the torch policy on a few frames.

---

## Phase 3 — T1 motion dataset (small)

Goal: ~10% scale T1 dataset plus the T1 side of the WBT training config.

- **3.1** There's no `config_values/wbt/t1/` yet — create it, mirroring the
  G1 directory. Joint counts match but names/indices differ; ordered joint
  lists live in the robot config at
  `src/holosoma/holosoma/config_values/robot.py`.
- **3.2** Pick the T1 clip subset (~10% of G1 dataset). Critical: draw from
  the **same motion-type distribution** as G1 so the transfer story is
  "less data, same distribution," not "different distribution."
- **3.3** Run LAFAN→T1 retargeting (`("lafan", "t1")` mapping exists) and
  produce NPZ files.
- **3.4** Visual sanity check on T1 the same way as 1.4. T1 has different
  ankle geometry and may fail on clips G1 handled fine — note and log.
- **3.5** Train a **T1-from-scratch baseline** on this small set.
  This is one of your Phase 0.3 baselines; don't skip it.

---

## Phase 4 — Transfer harness (G1 → T1 finetuning)

Goal: infrastructure to load a G1 checkpoint, adapt the network for T1, and
continue training.

- **4.1** Design the adapter strategy per Phase 0.2. Concretely, that means
  deciding:
  - which parameter tensors to copy from G1 checkpoint
  - which to reinitialize
  - whether any layers should be frozen early in finetuning
- **4.2** Extend `train_agent.py` (or add a sibling entry point) with a
  `--finetune-from <ckpt>` flag that loads G1 weights into a T1-configured
  policy with the adapter strategy from 4.1.
  - Handle shape mismatches explicitly — don't silently skip layers.
  - Log which params were loaded vs. reinit-ed vs. frozen.
- **4.3** Handle value/critic network too. Common gotcha: copying the actor
  but not the critic leads to huge initial value errors and unstable early
  finetuning. Decide: reinit critic, copy critic, or warmup critic-only
  for N steps.
- **4.4** Handle observation/action normalizers. The G1 empirical
  normalization stats are wrong for T1's obs distribution. Reset or
  rescale them.
- **4.5** Write a config for the finetune run (smaller LR, possibly shorter
  horizon, KL-penalty to prior? decide).
- **4.6** End-to-end smoke test: load G1 ckpt, step the T1 env for a few
  iterations, confirm nothing crashes and early rewards are non-pathological.

---

## Phase 5 — RL algorithm comparison

Goal: run PPO / FastSAC / FPO on the transfer task and compare.

- **5.1 PPO baseline.** Already implemented at `holosoma/agents/ppo/`. Use
  as the reference algorithm throughout Phases 2 & 4. Tune once, freeze
  hyperparameters for the comparison.
- **5.2 FastSAC.** Already implemented at `holosoma/agents/fast_sac/`. Off-
  policy — finetuning dynamics may differ meaningfully from PPO, and SAC
  with a loaded actor may need replay-buffer warmup before updates begin.
- **5.3 FPO (stretch).** Flow-matching policy — not in the repo. Subtasks:
  - 5.3.1 Literature pass: pick a specific flow-matching RL formulation
    (there are several; they differ in how gradients flow through the
    sampling chain). Write a 1-page design doc before coding.
  - 5.3.2 Implement the policy module under `holosoma/agents/fpo/`,
    following the `base_algo` interface.
  - 5.3.3 Verify on a trivial env (e.g., a low-dim control task) *before*
    plugging into the humanoid pipeline. Debugging flow RL bugs inside a
    29-DOF humanoid env is painful.
  - 5.3.4 Run the G1→T1 finetune experiment with FPO.
  - Fallback: if FPO is unstable after a bounded effort (e.g. 2 weeks),
    ship PPO + FastSAC results and leave FPO as future work.
- **5.4** For each algorithm, run the data-fraction sweep from Phase 0.3
  with at least 3 seeds per point. Budget this — it's the most expensive
  step.

---

## Phase 6 — Evaluation and analysis

- **6.1** Compute Phase 0.4 metrics across all `(algorithm, data_fraction,
  seed)` runs. Aggregate into a results table / dataframe.
- **6.2** Produce the key plot: sample efficiency vs. data fraction, with
  and without G1 pretraining, per algorithm.
- **6.3** Qualitative: render comparison videos (scratch vs. transferred)
  on held-out T1 motion clips. Deploy best policy via `holosoma_inference/`
  on the real robot if hardware time is available.
- **6.4** Ablations worth considering: which layers to reinit (Phase 0.2
  variants), critic handling (Phase 4.3), dataset-distribution mismatch
  (train G1 on walking-only, transfer to T1 dancing — does the prior still
  help or hurt?).

---

## Risks and watch-items

- **G1/T1 observation semantics diverge.** Watch for silent dim-alignment
  bugs where a G1 joint index points to a semantically different T1 joint.
- **Retargeting quality ceiling.** Transfer results are capped by how
  physically plausible the retargeted motions are. Budget time to iterate
  on retargeting configs if tracking error plateaus high.
- **Critic transfer is underappreciated.** Several transfer-RL papers find
  critic reinit matters more than actor reinit. Worth an explicit ablation.
- **FPO scope creep.** Easiest way to sink the project. Timebox hard.
- **Reward hacking.** Motion-tracking policies love to exploit reward
  terms (e.g., satisfying pos error by standing still at a mean pose).
  Eyeball rollouts, don't trust reward curves alone.
