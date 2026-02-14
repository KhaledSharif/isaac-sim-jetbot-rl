# Off-Policy RL Migration Plan

## Why Switch from PPO to Off-Policy?

PPO (on-policy) has a fundamental tension with BC warmstart: the pretrained policy is tight (low std), so each gradient step produces large KL divergence, triggering early stopping and throttling learning. Off-policy methods avoid this entirely because they don't have a "trust region" constraint between the data collection policy and the learning policy.

**Current pipeline problems:**
- KL early stopping fires on step 0-3 of every PPO epoch (wastes 70-100% of gradient budget)
- On-policy data is expensive: each 2048-step rollout takes ~100s in Isaac Sim, then most of it gets thrown away
- Demo data can only be used once (for BC pretraining), not replayed continuously

## Recommended Architecture: SAC + RLPD

### What is RLPD?

RLPD (Reinforcement Learning with Prior Data) is a simple extension of SAC that prepopulates the replay buffer with demonstration transitions and oversamples them during training. It was shown to exceed prior SOTA on continuous control benchmarks while being much simpler than alternatives.

**Paper:** Ball et al., "Efficient Online Reinforcement Learning with Offline Data" (ICML 2023)
**Code:** https://github.com/ikostrikov/rlpd

### Key Design Choices

1. **SAC (Soft Actor-Critic)** as the base algorithm
   - Off-policy: reuses all past experience via replay buffer
   - Entropy-regularized: automatic exploration without manual std tuning
   - High sample efficiency: update-to-data ratio >> 1 (e.g., 20 gradient steps per env step)

2. **Demo replay buffer prepopulation**
   - Load all 37,645 demo transitions into the replay buffer at initialization
   - Each training batch samples 50% from demo data, 50% from online data
   - Demo data is reused continuously throughout training (not just at init)

3. **Key modifications from vanilla SAC** (per RLPD paper)
   - High UTD (update-to-data) ratio: 20 gradient steps per environment step
   - Layer normalization in critic networks (stabilizes high-UTD training)
   - Ensemble of 10 Q-functions (reduces overestimation with offline data)

### Alternative: IBRL (Higher Complexity, Higher Performance)

IBRL keeps a frozen BC policy alongside the RL policy. At each step, both propose an action and the one with higher Q-value is executed. This achieved 6.4x higher success than RLPD on sparse-reward tasks.

**Trade-off:** More complex to implement, but better for sparse rewards. Our dense reward setup may not need it.

## Implementation Plan

### Phase 1: SAC Baseline (No Demos)

**File:** `src/train_sac.py` (new)

```
Dependencies: stable-baselines3 (already installed, includes SAC)
```

1. Create SAC training script mirroring `train_rl.py` structure
2. Use same `JetbotNavigationEnv` (no env changes needed)
3. SAC hyperparameters for continuous navigation:
   - `learning_rate=3e-4`
   - `buffer_size=100_000`
   - `batch_size=256`
   - `tau=0.005` (soft target update)
   - `gamma=0.99`
   - `train_freq=1` (update every step)
   - `gradient_steps=1` (start conservative, increase later)
   - `learning_starts=1000` (random exploration warmup)
4. VecNormalize still needed for observation normalization
5. No reward normalization (SAC's entropy term handles scale)

### Phase 2: RLPD Demo Integration

1. Load demo NPZ into replay buffer before training starts:
   ```python
   # Pseudocode
   demo = np.load("demos/recording.npz")
   for i in range(len(demo['observations'])):
       replay_buffer.add(
           obs=demo['observations'][i],
           action=demo['actions'][i],
           reward=demo['rewards'][i],
           next_obs=demo['observations'][i+1],  # need to handle episode boundaries
           done=demo['dones'][i],
       )
   ```
2. Modify SB3's replay buffer sampling to oversample demo data (50/50 split)
   - Option A: Subclass `ReplayBuffer` with a `demo_ratio` parameter
   - Option B: Use two separate buffers and concatenate batches
3. Pre-warm VecNormalize obs_rms from demo data (same as current pipeline)

### Phase 3: High UTD + Stabilization

1. Increase `gradient_steps` to 20 (20 gradient updates per env step)
2. Add layer normalization to critic network:
   ```python
   policy_kwargs=dict(
       net_arch=dict(pi=[256, 256], qf=[256, 256]),
       use_layer_norm=True,  # May need custom policy class
   )
   ```
3. If overestimation is an issue, consider:
   - TQC (Truncated Quantile Critics) from sb3-contrib — ensemble Q-functions built in
   - `pip install sb3-contrib` then `from sb3_contrib import TQC`

### Phase 4: Evaluation & Comparison

1. Train SAC+RLPD for same wall-clock time as PPO
2. Compare:
   - Sample efficiency (success rate vs. env steps)
   - Wall-clock efficiency (success rate vs. time)
   - Final performance (success rate at convergence)
3. If SAC+RLPD wins, retire `train_rl.py` or keep as alternative

## Migration Checklist

- [ ] `src/train_sac.py` — SAC training script with same CLI interface as `train_rl.py`
- [ ] Demo replay buffer loading (handle episode boundaries for next_obs)
- [ ] 50/50 demo/online sampling ratio
- [ ] VecNormalize pre-warming (reuse existing `prewarm_vecnormalize()`)
- [ ] High UTD ratio with layer norm stabilization
- [ ] `eval_policy.py` compatibility (should work as-is since it loads any SB3 model)
- [ ] Update `CLAUDE.md` and `README.md` with SAC training commands
- [ ] TensorBoard logging comparison

## References

- RLPD: https://github.com/ikostrikov/rlpd (Ball et al., ICML 2023)
- IBRL: https://github.com/hengyuan-hu/ibrl (Hu et al., RSS 2024)
- S2E: https://arxiv.org/html/2507.22028 (BC pretrain + PPO fine-tune for navigation)
- SB3 SAC: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- SB3-Contrib TQC: https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html
