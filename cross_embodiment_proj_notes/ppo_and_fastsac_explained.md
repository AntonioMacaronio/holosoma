# PPO and FastSAC in this repo, explained

This is a "from policy gradient" walkthrough of how the two RL algorithms used
in the cross-embodiment project (see `plan.md`) are actually implemented in
`src/holosoma/holosoma/agents/`. It assumes you know the basic policy
gradient idea — that we want to push up the log-probability of actions that
turned out well — and fills in everything else.

The two algorithms play different roles in the project:

- **PPO** (`agents/ppo/ppo.py`) — on-policy, the workhorse. Used for the
  G1 pretraining run in Phase 2 and as the reference algorithm everywhere.
- **FastSAC** (`agents/fast_sac/`) — off-policy, sample-efficient,
  maximum-entropy. Used in Phase 5 as the second axis of comparison
  alongside PPO.

Both are actor-critic. Both train a stochastic policy (the **actor**) and a
value function (the **critic**) simultaneously. Where they differ is *how*
the gradient signal for the actor is built and *what data* they reuse.

---

## 0. Shared scaffolding

Before getting into the algorithm-specific stuff, here's what they have in
common in this repo:

- **Massively parallel envs.** The simulator runs `num_envs` (often 4096+)
  copies of the robot in parallel. Every "step" produces `num_envs`
  transitions at once. This is why both algorithms are designed around
  big-batch GPU training rather than single-env rollouts.
- **Asymmetric actor / critic observations.** The actor sees a
  proprioceptive observation (`actor_obs`) — what the real robot would
  sense. The critic sees a richer one (`critic_obs`) including privileged
  simulator state. The critic is only used at training time, so this
  doesn't break sim-to-real.
- **Empirical observation normalization.** A running mean/std normalizer
  (`EmpiricalNormalization` in `ppo.py:40` and the FastSAC equivalent)
  centers and scales obs before they hit the network. Important for
  stability but also a transfer footgun — see plan.md §4.4.
- **Stochastic Gaussian actor.** Both algorithms parameterize the policy
  as a multivariate Gaussian over actions: the network outputs a mean
  vector, and a per-dimension standard deviation. To act, sample from
  that Gaussian. To get gradients, use the log-probability of the
  sampled action.

That's the joint setup. Now the algorithms.

---

## 1. PPO — Proximal Policy Optimization

### 1.1 The intuition

Vanilla policy gradient (REINFORCE) says: increase the log-prob of actions
that led to high return, decrease it for actions that led to low return.
Concretely, if `A(s,a)` is the *advantage* of taking action `a` in state
`s` (how much better than average), the gradient is

```
∇θ J = E[ A(s,a) · ∇θ log πθ(a|s) ]
```

Two practical problems with this:

1. **High variance.** A single-trajectory return is a very noisy estimate
   of the true expected return. → fix with a **value-function critic**:
   subtract a learned baseline and use temporal-difference returns.
2. **Big policy updates blow up.** If you take a large gradient step, the
   new policy is so different from the one that collected the data that
   the data is no longer informative — you can crater performance in one
   update. → fix with **PPO's clipped objective**: take many small steps
   on the same batch of data, but cap how far each one can move the
   policy.

### 1.2 The PPO objective, demystified

For each transition `(s, a, A)` we collected with the *old* policy
`πθ_old`, define the *importance ratio*

```
r(θ) = πθ(a|s) / πθ_old(a|s)
```

This is the ratio of probabilities of the *same action* under the new vs.
old policy. If `r = 1`, no change. The naive policy-gradient surrogate is
just `r · A`. PPO clips `r` to `[1 - ε, 1 + ε]` (the `clip_param`, here
`ε ≈ 0.2`):

```
L_surr = -mean( min( r · A,  clip(r, 1-ε, 1+ε) · A ) )
```

Translated: **as long as the new policy isn't much different from the old
one, this looks like normal policy gradient. The moment we try to push the
ratio outside `[1-ε, 1+ε]` in a way that would increase reward, the
gradient zeroes out**, capping the update size. This lets us do multiple
SGD passes over the same batch (`num_learning_epochs`) without diverging.

You can see this exact computation in `ppo.py:565-570`:

```python
ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
surrogate = -advantages * ratio
surrogate_clipped = -advantages * torch.clamp(ratio, 1-ε, 1+ε)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

(The `max` rather than `min` is because the loss is the *negative* of the
objective.)

### 1.3 Where the advantage `A` comes from: GAE

The advantage `A(s,a)` is *how much better than the average* a particular
action was. We don't have it for free — we estimate it from collected
returns and the critic's value estimates `V(s)`. PPO uses **Generalized
Advantage Estimation (GAE)**, a weighted blend of n-step TD residuals
controlled by `λ` (`config.lam`, usually 0.95):

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)        # 1-step TD error
A_t = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

Small λ ≈ low variance, more biased (closer to 1-step TD). Large λ ≈ high
variance, less biased (closer to Monte Carlo return). `λ=0.95` is the
common sweet spot. The GAE recursion is implemented in
`ppo.py:_compute_returns_and_advantages` (lines 452-472), iterating
backward through the rollout. After computing, advantages are normalized
to mean 0 / std 1 for stability.

### 1.4 The critic loss

The critic is trained to predict the GAE returns by squared error.
PPO additionally clips the value update so a freshly updated critic
can't move too far either (`ppo.py:573-578`):

```python
value_clipped = old_values + (V_new - old_values).clamp(-ε, ε)
value_loss = max((V_new - returns)², (value_clipped - returns)²).mean()
```

### 1.5 Entropy bonus

The actor loss adds a small entropy bonus
(`-entropy_coef · H(π)`) to keep the policy from collapsing
prematurely to a deterministic action. Concretely you can see it in
`ppo.py:603-608`:

```python
actor_loss = surrogate_loss - entropy_coef * entropy + ...
```

### 1.6 Adaptive learning rate from KL

Instead of hand-tuning the learning rate, PPO here measures the empirical
KL divergence between old and new policies after each update and adapts
the LR (`ppo.py:_update_learning_rate`, lines 637-648):

- If KL is too large (> 2× target) → halve the LR (×1/1.5).
- If KL is too small (< 0.5× target) → grow the LR (×1.5).

Target KL is `desired_kl ≈ 0.01`. This is a really useful trick — it
keeps each update at roughly the same "policy distance" regardless of
where you are in training.

### 1.7 The full PPO loop (one iteration)

This is what `PPO.learn()` does each iteration (`ppo.py:346-386`):

1. **Rollout** (`_rollout_step`, line 387): step the env `num_steps_per_env`
   times in parallel across all `num_envs`. Store
   `(actor_obs, critic_obs, action, log_prob, value, reward, done)` for
   every transition.
2. **Compute returns and advantages** with GAE over the collected batch.
3. **Update** (`_training_step`, line 474): for `num_learning_epochs`,
   shuffle into `num_mini_batches`, compute `actor_loss + critic_loss`,
   backprop, clip grad norm, step both optimizers. Optionally adapt LR
   based on observed KL.
4. **Clear the buffer.** All data is on-policy: it was collected by the
   policy you just updated *off of*, so it's now stale. Throw it away
   and rollout again.

The on-policy property is what makes PPO sample-inefficient compared to
SAC — every batch gets used for `num_learning_epochs × num_mini_batches`
gradient steps, then discarded. But each rollout is fast (parallel envs)
and the algorithm is famously stable. That's why it's the default for
sim-trained robotics.

---

## 2. FastSAC — fast Soft Actor-Critic

FastSAC is a heavily optimized variant of SAC adapted for massively
parallel sim training (the `agents/fast_sac/` docstring at
`fast_sac_agent.py:147-153` cites the FastTD3 paper, arXiv:2505.22642,
which uses the same recipe). The high-level shape is **off-policy
maximum-entropy actor-critic**. Three things to internalize first:

### 2.1 Off-policy + replay buffer

PPO collects data, updates, throws it away. SAC keeps every transition
in a **replay buffer** (`SimpleReplayBuffer` in
`fast_sac/fast_sac_utils.py:15`) and trains by sampling random batches
from it — even transitions collected by a much older policy are still
useful, because the critic update is off-policy (Bellman backup, not
policy gradient on collected returns). This is *the* reason SAC tends to
be more sample-efficient than PPO: each transition is reused many times.

The cost: you need a critic that's accurate off-policy, which means
TD-style updates with target networks (more on that below).

### 2.2 Maximum-entropy objective

Standard RL maximizes expected return `E[Σ r_t]`. SAC instead maximizes

```
J(π) = E[ Σ ( r_t + α · H(π(·|s_t)) ) ]
```

where `H(π)` is the policy's entropy and `α` is a temperature. In words:
**reward yourself for being random.** Entropy regularization makes
exploration automatic, smooths the value function, and prevents premature
collapse to a deterministic policy.

The entropy term shows up in two places:

1. **In the critic target** (`fast_sac_agent.py:433`): the bootstrapped
   target value subtracts `α · log π(a'|s')`, so the critic learns the
   *soft* Q-value, including the bonus for stochastic action selection.
2. **In the actor loss** (`fast_sac_agent.py:511`): minimize
   `α · log π(a|s) - Q(s,a)`, which is equivalent to maximizing
   `Q(s,a) - α · log π(a|s)` — pull toward high-Q actions, but with a
   pull-toward-uniform regularizer.

`α` is auto-tuned (see 2.6).

### 2.3 The actor: tanh-squashed Gaussian with reparametrization

The actor (`fast_sac.py:8-156`) outputs a Gaussian over a "raw" action
`u ~ N(μ(s), σ(s))`, then squashes it through `tanh` and scales it
per-joint:

```
action = tanh(u) · action_scale + action_bias
```

The per-joint `action_scale` (computed in
`fast_sac_agent.py:_compute_action_boundaries`) ensures `tanh(u) = ±1`
maps to the joint limit furthest from the default pose. So `action = 0`
means "stay at default," `action = ±1` means "go to the joint limit."

The reason for `tanh` rather than just sampling from a Normal: bounded
actions matter for hardware (joint limits), and `tanh` keeps gradients
well-behaved. The catch is that you have to correct the log-probability
for the change-of-variables — the Jacobian term `log(1 - tanh(u)²)`
appears in `fast_sac.py:114`. This is required for the actor loss to be
mathematically correct.

The sample is drawn with **rsample**
(`get_actions_and_log_probs`, line 99), i.e. the *reparametrization
trick*: `u = μ + σ · ε` where `ε ~ N(0, I)` is fixed noise, so gradients
flow through `μ` and `σ` directly. This is what makes the SAC actor
update a low-variance pathwise gradient (not a REINFORCE-style
log-prob-weighted score-function gradient). It's a meaningful difference
from PPO: PPO uses the score-function estimator (sample, then weight by
log-prob); SAC uses the pathwise estimator (sample = function of params,
backprop through it).

The mean and log_std heads are zero-initialized (`fast_sac.py:76-79`), so
at step 0 the policy is "centered Gaussian, near default pose with
moderate noise." Useful for not spazzing out before learning starts.

### 2.4 The critic: distributional twin Q-networks

This is where FastSAC departs from textbook SAC. Two things going on:

**Twin Q-networks** (`num_q_networks=2` by default). Two independent Q
networks `Q1, Q2` are trained on the same data. When computing the actor
loss / value, take the *minimum* (or in this implementation an average
over Q-heads via `mean(dim=0)` — see `fast_sac_agent.py:510`). The
purpose is to fight Q-value *overestimation bias*: if a single Q network
overestimates some action, the actor will try to exploit that
overestimate, and the bias compounds. Two networks act as a sanity
check.

**Distributional / categorical Q.** Instead of regressing to a scalar
Q-value, each Q-network outputs a *distribution* over a fixed support
`[v_min, v_max]` discretized into `num_atoms` (e.g. 51) bins — this is
the C51 trick. So `Q(s,a)` is represented as a softmax over those bins,
and the predicted value is `Σ p_i · z_i` (`Critic.get_value`,
`fast_sac.py:382`).

The Bellman update becomes a **distribution projection**: take the
target distribution at `(s', a')`, shift it by the reward, contract by
γ, then project back onto the fixed support. That's what
`DistributionalQNetwork.projection` (`fast_sac.py:245-290`) computes.
The critic loss is the cross-entropy between the projected target
distribution and the predicted distribution
(`fast_sac_agent.py:441-444`).

Why distributional? Empirically, predicting the full return distribution
gives a better gradient signal than predicting just the mean — partly
because the cross-entropy loss has nicer optimization properties than
MSE, partly because it captures multimodality in returns.

### 2.5 Target networks and soft updates

The critic target uses a *target network* `qnet_target` whose weights
slowly track the live `qnet` via Polyak averaging (`tau` ≈ 0.005):

```
θ_target ← (1 - τ) · θ_target + τ · θ_live
```

See `fast_sac_agent.py:770-774`. Without this, the critic target moves
every gradient step and you get unstable bootstrapping. The slow target
is the standard fix in DQN-family algorithms.

### 2.6 Auto-tuned entropy temperature `α`

The temperature `α` controls the entropy/return tradeoff. Hand-tuning it
is brittle (different reward scales need different α). SAC instead sets
a *target entropy* `H_target = -n_act · target_entropy_ratio`, and
trains `log α` by gradient descent against:

```
L(α) = -α · ( log π(a|s) + H_target )
```

(See `fast_sac_agent.py:466`.) Reading the gradient: if current entropy
> target, push α down (less entropy regularization needed); if current
entropy < target, push α up. Net effect: entropy auto-stabilizes near
target.

### 2.7 The full FastSAC loop (one global step)

Each "global step" (`FastSACAgent.learn`, `fast_sac_agent.py:650-811`)
does roughly:

1. **Collect** one env step in parallel: query the actor for actions on
   current obs, step the env, push the transition into the replay
   buffer.
2. **Skip until warmup is done.** For the first `learning_starts` global
   steps, just collect — no gradient updates. This fills the buffer
   with diverse data before training starts. (Plan.md §5.2 calls this
   out as a finetuning concern.)
3. **Sample from the buffer.** A big batch (`batch_size × num_updates`)
   is drawn from the replay buffer once, normalized once, and split
   into `num_updates` smaller batches. This amortizes sampling and
   normalization cost (`_sample_and_prepare_batches`, line 537).
4. **For each minibatch:**
   - **Critic update** (`_update_main`, line 404): compute target
     distribution from the target network and a fresh actor sample,
     project, compute cross-entropy critic loss, backprop, optimizer
     step. Update α by its loss too.
   - **Actor update** (`_update_pol`, line 489), every
     `policy_frequency` updates: actor loss is
     `α · log π(a|s) - Q(s,a)`, both with rsampled `a`. Backprop
     through the actor (the pathwise gradient — see 2.3).
   - **Soft-update the target network** by Polyak.
5. **Loop.**

Compared to PPO, every "iteration" is much cheaper (one env step, one
actor update — vs. PPO's `num_steps_per_env × num_envs` rollout +
`num_epochs × num_mini_batches` of optimization). But you need to do
many more iterations because each contains less new data. The net
sample efficiency is usually better than PPO; the wall-clock efficiency
is more nuanced and depends on simulation speed vs. update speed.

### 2.8 Other production-grade details

A few engineering knobs worth knowing about, since they show up in the
config:

- **AMP (`amp=True`).** Mixed-precision (bf16 or fp16) forward + loss
  for speed. See `_maybe_amp` (`fast_sac_agent.py:358`).
- **`torch.compile`.** Optional graph compilation of the update step
  for further speedup (`fast_sac_agent.py:653-658`).
- **LayerNorm + SiLU.** The MLP trunks use LayerNorm between linear
  layers and SiLU activations (vs. PPO's plain ReLU/ELU). Both
  empirically help with off-policy stability.

---

## 3. So how do they actually compare?

For someone with intro RL background, the cleanest way to remember the
contrast:

| Aspect | PPO | FastSAC |
|---|---|---|
| On/off policy | On-policy (data thrown away) | Off-policy (replay buffer) |
| Actor gradient | Score function (`A · ∇log π`), clipped | Pathwise (`∇Q ∘ rsample`) |
| Critic loss | MSE to GAE returns, clipped | Cross-entropy on projected distribution |
| Q networks | One value head | Two distributional Q heads |
| Exploration | Entropy bonus | Maximum-entropy objective + auto-α |
| Stability trick | Clipped surrogate, KL-adaptive LR | Twin Q + slow target net |
| Sample efficiency | Lower | Higher |
| Wall-clock w/ parallel sim | Excellent | Good (depends on update cost) |

For the project specifically (plan.md §5), PPO is the reliable workhorse
that gets the G1 pretrain done, and FastSAC is the sample-efficient
comparator that's expected to shine when T1 data is scarce — which is
exactly the regime the data-fraction sweep targets.

---

## 4. Where to read the code

Quick file map if you want to dig further:

- `agents/ppo/ppo.py` — full PPO implementation, ~900 lines, all in one
  class. Start at `learn()` (line 346) and follow `_rollout_step` →
  `_training_step` → `_update_algo_step` → `_compute_ppo_loss`.
- `agents/fast_sac/fast_sac.py` — actor and critic network classes
  (Gaussian-tanh actor, distributional Q with C51 projection).
- `agents/fast_sac/fast_sac_agent.py` — the FastSAC training loop.
  Start at `learn()` (line 650) and follow `_update_main` (critic +
  alpha) and `_update_pol` (actor).
- `agents/fast_sac/fast_sac_utils.py` — `SimpleReplayBuffer`,
  `EmpiricalNormalization`, checkpoint helpers.
- `agents/modules/modules.py:131` — `build_mlp_layer`, the shared
  PPO trunk builder. (FastSAC builds its trunks inline rather than
  through this helper — relevant for plan.md §2.2.)
