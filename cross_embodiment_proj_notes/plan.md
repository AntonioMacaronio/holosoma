# Cross-Embodiment Policy Transfer: G1 → T1

Working breakdown of the research pipeline. Phases are roughly sequential, but
items within a phase can often be parallelized. Open decisions are flagged
inline — resolve them before the phase that depends on them.

---

## Phase 0 — Decisions (locked in)

- **0.1 Task framing: whole-body motion tracking (WBT).** Policy conditioned
  on future reference frames. Existing infra in
  `src/holosoma/holosoma/config_values/wbt/g1/` is the right starting point.
- **0.2 Transfer strategy: 5-layer MLP, reinit first and last layers.**
  Pretrain the G1 policy as a 5-Linear-layer MLP (one deeper than the
  repo default of 4 Linear layers) so the middle has capacity to encode
  an embodiment-agnostic skill representation. On T1 finetune,
  reinitialize layer 1 (obs → hidden) and layer 5 (hidden → action)
  while keeping layers 2/3/4 from the G1 checkpoint.
  - **PPO width**: `hidden_dims=[512, 512, 256, 128]` (current repo
    default is `[512, 256, 128]` at `config_values/algo.py:46,52`).
    `build_mlp_layer` in `agents/modules/modules.py:131` turns this into
    5 Linear layers:
    ```
    Layer 1: obs → 512   [reinit on T1]
    Layer 2: 512 → 512   [keep from G1]
    Layer 3: 512 → 256   [keep from G1]
    Layer 4: 256 → 128   [keep from G1]
    Layer 5: 128 → act   [reinit on T1]
    ```
  - **FastSAC**: depth is *hardcoded* in `agents/fast_sac/fast_sac.py`
    (4 Linear layers, widths `h → h/2 → h/4 → act` for the actor). Edit
    `fast_sac.py` to insert a second `h → h` layer at the front so both
    algorithms use 5 Linear layers with the same middle-3 transfer
    pattern. See Phase 2.2 for implementation notes.
  - Apply the same reinit-first-and-last scheme to the critic — see 4.3.
- **0.3 Baselines: sweep over dataset fraction.** For each algorithm, run
  `{1, 5, 10, 25, 50, 100}%` of the T1 dataset, both from-scratch and
  G1-finetuned. 3 seeds per point. Plot as learning-efficiency curves.
- **0.4 Eval metrics (three):**
  - (a) motion tracking error (per-frame joint/body pose error vs. reference)
  - (b) torso velocity tracking (commanded vs. achieved root velocity)
  - (c) locomotion stability — survival time / fall rate
- **0.5 Algorithms: PPO and FastSAC only.** No FPO. Comparison axes are
  `{algorithm} × {scratch vs. finetuned} × {data fraction}`.

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
- **2.2** Set actor and critic architecture to a 5-Linear-layer MLP
  (per 0.2). Specifics per algorithm:
  - **PPO**: config-only change. In `config_values/algo.py:46,52` set
    `hidden_dims=[512, 512, 256, 128]` for both actor and critic.
    Depth is driven by the list length via `build_mlp_layer` in
    `agents/modules/modules.py:131`. Can be overridden per-experiment
    in `config_values/wbt/g1/experiment.py` via `replace()` instead of
    touching the global default if you want to keep other experiments
    on the old size.
  - **FastSAC**: code change in `agents/fast_sac/fast_sac.py`. The actor
    (around line 62) and critic (around line 225) each define a
    4-Linear-layer Sequential with hardcoded widths `h → h/2 → h/4 → out`.
    Insert an extra `Linear(h, h) → activation → LayerNorm?` block right
    after the first layer to make it 5 Linear layers: `h → h → h/2 → h/4 → out`.
    Match whatever normalization/activation pattern the existing layers
    use (check for LayerNorm since `use_layer_norm=True` in the config).
    Widths `actor_hidden_dim=512`, `critic_hidden_dim=768` stay as-is.
  - Hidden-width contract: the layer-2/3/4 shapes (512→512, 512→256,
    256→128 for PPO) are the contract T1 must match for checkpoint load
    to succeed.
- **2.3** Wire the Phase 1 dataset into the command manager. Confirm clip
  sampling, looping, and reset behavior.
- **2.4** Short smoke-train on a single clip to verify the pipeline
  end-to-end before committing GPU hours.
- **2.5** Full PPO training run on the G1 dataset using `train_agent.py`.
  Log to W&B. Save checkpoints.
- **2.6** Evaluate with `eval_agent.py` on held-out clips. Record the
  Phase 0.4 metrics — these are your "upper-bound" reference numbers.
- **2.7** Export to ONNX (existing infra in `holosoma_inference/`).
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

Goal: infrastructure to load a G1 checkpoint into a T1-configured 5-layer
MLP, reinitializing layers 1 and 5, and continue training.

- **4.1 Architecture.** T1 policy is the same 5-layer MLP as G1 (same
  hidden width — locked in Phase 2.2). Layer 1 input dim is `t1_obs_dim`;
  layer 5 output dim is `t1_act_dim`. Layers 2/3/4 are bit-identical in
  shape to G1.
  - Read `config_values/wbt/g1/observation.py` and the Phase 3.1 T1 obs
    config to confirm obs dims. Action dims from `config_values/robot.py`.
  - Initialization for reinit layers: use the same init the repo uses for
    scratch training (likely orthogonal or small-gain default in
    `holosoma/agents/modules/`). No need for near-zero-init — the middle
    layers already carry the G1 prior and layer 5 must learn to route
    into T1's joint order from scratch anyway.
- **4.2 Checkpoint loader.** Extend `train_agent.py` (or add a sibling
  entry point) with a `--finetune-from <ckpt>` flag that:
  - loads layers 2, 3, 4 (both weights and biases) from the G1 checkpoint,
  - freshly initializes layers 1 and 5 per repo defaults,
  - logs which params were loaded vs. reinitialized,
  - errors loudly on unexpected shape mismatches on the middle layers —
    don't silently skip (that would mean the G1 and T1 hidden widths
    drifted apart, which invalidates the whole transfer).
- **4.3 Critic handling.** Apply the same reinit-first-and-last scheme
  to the critic: load layers 2/3/4 from the G1 critic, reinit layers 1
  and 5. Rationale: T1 observations feed a different first layer, and
  the value scale on T1 may differ anyway — a fresh head is safer than
  a fresh-obs-to-stale-head pairing. Consider critic-only warmup (freeze
  actor for N steps) if early PPO/SAC updates look unstable.
- **4.4 Normalizers.** G1's empirical obs/action normalization stats are
  wrong for T1. Reset them and let them re-accumulate from T1 rollouts,
  or collect a small T1 warmup buffer to seed them before the first
  policy update.
- **4.5 Finetune hyperparameters.** Likely smaller LR than scratch
  training; consider a KL penalty or trust-region against the G1 prior
  for the first few updates to avoid wrecking transferred middle layers.
  Freeze experiment — don't re-tune per data fraction.
- **4.6 Freezing schedule (optional ablation).** Worth trying: freeze
  middle layers (2/3/4) for the first K steps (train only the reinit-ed
  layers 1 and 5), then unfreeze everything. Cheap to add, sometimes
  meaningfully more stable.
- **4.7 Smoke test.** Load G1 ckpt into T1 policy with layers 1 and 5
  reinit-ed, step the T1 env, confirm no crashes and that early rewards
  are not pathological. Note: unlike the adapter-with-near-zero-init
  variant, the T1 policy at step 0 will produce near-random T1 actions
  (because layer 5 is random), so immediate falls are expected — the
  question is whether learning recovers within the first few iterations.

---

## Phase 5 — RL algorithm comparison (PPO vs. FastSAC)

Goal: run PPO and FastSAC across scratch / finetuned × data-fraction and
compare on the Phase 0.4 metrics.

- **5.1 PPO.** Already implemented at `holosoma/agents/ppo/`. Reference
  algorithm throughout Phases 2 & 4. Tune once on G1 pretraining
  (Phase 2), freeze hyperparameters for the comparison sweep.
- **5.2 FastSAC.** Already implemented at `holosoma/agents/fast_sac/`.
  Off-policy — watch for these differences vs. PPO when finetuning:
  - Replay buffer starts empty; may want a T1 rollout warmup before
    updates begin, otherwise early gradients are dominated by actor-loss
    on near-untrained critic.
  - SAC's entropy target is a meaningful knob during finetuning — too
    high and it immediately randomizes away from the G1 prior.
- **5.3 Sweep grid.** For each
  `(algorithm ∈ {PPO, FastSAC}) × (mode ∈ {scratch, finetune}) × (frac ∈ {1, 5, 10, 25, 50, 100}%)`
  run ≥3 seeds. That's 2×2×6×3 = **72 runs**. Budget compute and storage
  accordingly; stagger so you see scratch-vs-finetune at 10% early for a
  quick sanity signal before committing to the full grid.
- **5.4 Hyperparameter fairness.** PPO and FastSAC can't share hyperparams
  literally, but do freeze each algorithm's hyperparams across the sweep
  so the only independent variable is `(mode, frac)`. Document what you
  tuned and where.

---

## Phase 6 — Evaluation and analysis

- **6.1** Compute the three Phase 0.4 metrics — (a) motion tracking error,
  (b) torso velocity tracking error, (c) locomotion stability (survival
  time / fall rate) — across all `(algorithm, mode, data_fraction, seed)`
  runs on a held-out T1 clip set. Aggregate into one dataframe.
- **6.2** Key plots:
  - per-metric learning curve: metric vs. data fraction, with separate
    lines for scratch vs. finetuned, one subplot per algorithm
  - same grid but for sample efficiency (env steps to threshold)
- **6.3** Qualitative: render comparison videos (scratch vs. finetuned) on
  held-out T1 clips. Deploy best finetuned policy via `holosoma_inference/`
  on real T1 if hardware time is available.
- **6.4** Ablations worth considering (cheap wins, do if time permits):
  - critic handling variants (Phase 4.3)
  - freezing schedule (Phase 4.6)
  - distribution mismatch: pretrain G1 on walking only, transfer to T1
    dancing — does the prior still help, or hurt?

We actually ended up using these metrics for Table 1:
Motion Error = Env/motion/error_body_pos 
Velocity Error = Env/motion/error_ref_lin_vel
Avg Episode Length =  Env/average_episode_length
---

## Risks and watch-items

- **G1/T1 observation semantics diverge.** Watch for silent dim-alignment
  bugs where a G1 joint index points to a semantically different T1 joint.
- **Retargeting quality ceiling.** Transfer results are capped by how
  physically plausible the retargeted motions are. Budget time to iterate
  on retargeting configs if tracking error plateaus high.
- **Critic transfer is underappreciated.** Several transfer-RL papers find
  critic reinit matters more than actor reinit. Worth an explicit ablation.
- **Reward hacking.** Motion-tracking policies love to exploit reward
  terms (e.g., satisfying pos error by standing still at a mean pose).
  Eyeball rollouts, don't trust reward curves alone.
- **Cold-start falls.** Layer 5 is randomly initialized, so the T1 policy
  at step 0 outputs near-random actions and the robot will fall
  immediately. That's expected — the signal to watch is whether reward
  curves recover within the first few iterations rather than diverging.
- **Middle-layer clobbering.** The gradient flowing into layers 2/3/4
  from a freshly-random layer 5 can be large and noisy in the first few
  updates, which may damage the pretrained representations before they
  get useful gradients. The freezing schedule in 4.6 is the main
  mitigation; the KL-to-prior penalty in 4.5 is the backup.
- **Hidden-width drift.** If G1 pretrain hidden width ≠ T1 finetune
  hidden width, the middle-layer load fails. The shape-mismatch check
  in 4.2 exists specifically to catch this early instead of silently.
- **Sweep cost.** 72 runs × single-seed training time can be weeks of
  GPU-hours. Stage the sweep: get one `(algo, mode, 10%)` cell fully
  working end-to-end before fanning out.
