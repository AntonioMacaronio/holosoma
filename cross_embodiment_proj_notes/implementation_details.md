# Paper-ready implementation details

Drop-in paragraphs for the Implementation Details section. All numbers come
straight from the repository configs (cited inline). Where a number is intended
to be a free variable in our cross-embodiment study, it's marked **(swept)**.

## Simulator and environment

Training is performed in NVIDIA Isaac Sim 5.1 / Isaac Lab v2.3.0. We run 4096
parallel environments per training run on a single NVIDIA L40S GPU
(`config_values/wbt/g1/experiment.py:23`). Physics is stepped at 200 Hz with a
control decimation of 4 (`config_values/simulator.py:35–37`), giving an effective
policy control rate of **50 Hz**, matched to the LAFAN reference motion
playback rate. Episode length is capped at **10 s** (500 policy steps;
`experiment.py:50`). Default robot height at reset is 0.76 m for the Unitree G1
and 1.20 m for the Booster T1 (`config_types/robot.py` defaults). Self-collisions
are enabled (`experiment.py:60`). Observation and action terms follow the
manager-style configuration in `config_values/wbt/g1/{observation,action}.py`,
with action scale 0.25 applied per-joint and rescaled by effort-limit-over-PD-gain
(`experiment.py:57–58`).

## Policy architecture

We use a 5-layer fully-connected MLP for both the actor and the critic, with
hidden widths `[512, 512, 256, 128]` and ELU activations
(`config_values/wbt/g1/experiment.py:21–39`, `config_values/algo.py:46`,
`agents/modules/modules.py:131`). This deepens the repository's
4-layer default by inserting a `512 → 512` block at the front so the middle
three layers can absorb an embodiment-agnostic representation that is shared
between the G1 pretraining and T1 finetuning runs. For finetuning experiments,
layers 1 and 5 are reinitialized while layers 2/3/4 are loaded bit-shape-identical
from the G1 checkpoint.

## Reward function

Reward weights are listed in `config_values/wbt/g1/reward.py`. We compose six
exponential-shaped tracking rewards and three regularizers, all summed each
control step. Tracking rewards use `r = exp(−||x − x_ref||² / σ²)` with the
sigmas given below.

| Term | σ | Weight (PPO) | Weight (FastSAC) |
|---|---:|---:|---:|
| Global ref-body position error | 0.30 m | +0.5 | +1.0 |
| Global ref-body orientation error | 0.40 rad | +0.5 | +0.5 |
| Relative body position error | 0.30 m | +1.0 | +2.0 |
| Relative body orientation error | 0.40 rad | +1.0 | +1.0 |
| Body linear velocity error | 1.00 m/s | +1.0 | +1.0 |
| Body angular velocity error | 3.14 rad/s | +1.0 | +1.0 |
| Action-rate L2 penalty | — | −0.1 | −1.0 |
| DOF position limit violation (soft 0.9) | — | −10.0 | −10.0 |
| Undesired contacts (excluding feet/wrists) | — | −0.1 | −0.1 |

Tracking targets are derived per-frame from the LAFAN-retargeted reference
motion. The tracked-body set covers fourteen links: pelvis, hip-roll, knee, and
ankle on both legs; torso; and shoulder-roll, elbow, and wrist-yaw on both arms
(`config_values/wbt/g1/command.py:18–34`).

## Domain randomization

All randomizations live in `config_values/wbt/g1/randomization.py`.

**At setup** (per-environment, sampled once at world spawn):
- Rigid-body friction: static ∈ [0.30, 1.60], dynamic ∈ [0.30, 1.20], restitution ∈ [0.0, 0.5].
- Base center-of-mass offset: x ∈ ±0.025 m, y ∈ ±0.05 m, z ∈ ±0.05 m.
- DOF position bias: ±0.01 rad per joint (representing actuator zero-offset error).

**At reset** (per-episode):
- Initial pose around the LAFAN reference frame, with noise scales
  `dof_pos=0.1`, `root_pos=[0.05, 0.05, 0.01] m`, `root_rot=[0.10, 0.10, 0.20] rad`,
  `root_lin_vel=[0.5, 0.5, 0.2] m/s`, `root_ang_vel=[0.52, 0.52, 0.78] rad/s`,
  scaled uniformly by `overall_noise_scale=1.0` (`command.py:7–15`).
- DOF velocity randomization disabled; DOF position scale fixed at 1.0.
- Push schedule resampled.

**At each step**:
- Random external pushes applied at intervals sampled from [1, 3] s, with
  per-axis maximum velocity perturbations `[0.5, 0.5, 0.2] m/s` (linear) and
  `[0.52, 0.52, 0.78] rad/s` (angular).
- Actuator P/D-gain randomization (`kp_range=[0.9, 1.1]`, `kd_range=[0.9, 1.1]`)
  is wired but disabled by default for the WBT preset; we leave it off in our
  reported runs to keep the policy comparison clean.

## PPO

Hyperparameters from `config_values/algo.py:12–55` overridden by
`config_values/wbt/g1/experiment.py:25–41`:

- Optimizer: AdamW (separate actor and critic), `weight_decay = 0`,
  `learning_rate = 1 × 10⁻³` for both, max grad norm 1.0.
- Clip parameter ε = 0.2; KL-adaptive learning-rate schedule with target KL = 0.01.
- Discount γ = 0.99, GAE λ = 0.95.
- Value loss coefficient 1.0, entropy coefficient 0.005.
- Initial action noise std = 1.0; symmetry loss disabled
  (`use_symmetry=False`).
- Rollout: `num_steps_per_env = 24` × `num_envs = 4096` ⇒ **98,304 transitions per
  iteration**. Minibatches per epoch: 4 (≈ 24,576 transitions per minibatch).
  Epochs per iteration: 5. Effective minibatch size ≈ 24,576.
- Empirical observation normalization is enabled.
- Total length: **30,000 PPO iterations** (≈ 2.95 × 10⁹ environment transitions
  in total) for the G1 pretraining run; checkpoints saved every 4,000
  iterations. T1 from-scratch and finetuning runs use the same hyperparameters
  but only **(swept)** `num_learning_iterations` per data fraction (Phase 5
  budget).

## FastSAC

Hyperparameters from `config_values/algo.py:58–101` overridden by
`config_values/wbt/g1/experiment.py:84–146`:

- Optimizer: AdamW with `weight_decay = 1 × 10⁻³`. Actor LR, critic LR, and
  α LR all `3 × 10⁻⁴`. Mixed-precision training in bf16 (`amp=True`,
  `amp_dtype="bf16"`).
- Discount γ = 0.99, soft target update τ = 0.05.
- Replay buffer of 1024 most-recent rollouts; `num_steps = 1` per environment
  per iteration; **8192**-sample minibatch.
- Off-policy updates: `num_updates = 4` gradient steps per environment step,
  with `policy_frequency = 2` (actor updated once per two critic updates).
- Twin Q-networks (`num_q_networks = 2`); distributional value head with
  **501 atoms** spanning `[−20, 20]`.
- Entropy: target ratio 0.5; α auto-tuning enabled, initial α = 1 × 10⁻³;
  log-std clamped to `[−5, 0]`.
- Layer normalization in the actor and critic (`use_layer_norm = True`);
  `tanh`-squashed Gaussian outputs.
- Total length: **400,000 FastSAC iterations** for G1 pretraining; checkpoints
  every 1,000 iterations. T1 runs use the same hyperparameters but **(swept)**
  total iterations per data fraction.

## Logging and evaluation

Training metrics (mean episode reward, motion error per body, root linear and
angular velocity errors, average episode length, and PPO/FastSAC loss
components) are logged every iteration to Weights & Biases under the project
`WholeBodyTracking` (`experiment.py:21`). Evaluation rolls out the policy on
held-out LAFAN clips (3 of the 16 walk/run clips, stratified seed=0, 20 % val
fraction) and reports (a) mean per-body workspace position error, (b) mean
per-body joint position error, (c) commanded-vs-achieved root linear velocity
error, and (d) average episode length (survival time, capped at 500 policy
steps = 10 s).
