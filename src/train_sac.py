#!/usr/bin/env python3
"""Train SAC/TQC agent with RLPD-style demo replay for Jetbot navigation task.

Uses demonstrations in a 50/50 replay buffer (RLPD) instead of the fragile
BC warmstart → VecNormalize pre-warming pipeline. LayerNorm in critics replaces
VecNormalize entirely.

Usage:
    ./run.sh train_sac.py --demos demos/recording.npz --headless
    ./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 500000
    ./run.sh train_sac.py --demos demos/recording.npz --headless --cpu --timesteps 1000
"""

import argparse
import numpy as np
from pathlib import Path

from demo_utils import validate_demo_data, load_demo_transitions, VerboseEpisodeCallback


def symlog(x):
    """DreamerV3 symmetric log compression: sign(x) * log(|x| + 1)."""
    import torch
    return torch.sign(x) * torch.log1p(torch.abs(x))


class ChunkCVAEFeatureExtractor:
    """SB3 feature extractor for Chunk CVAE with split state/lidar MLPs.

    Splits 34D obs into state (0:10) and LiDAR (10:34), applies symlog,
    processes through separate MLPs, and concatenates with a zero-padded
    z-slot for the CVAE latent variable.

    Output: concat(state_features, lidar_features, z_pad) = 96 + z_dim
    """

    _cls = None

    @staticmethod
    def create(base_extractor_cls, z_dim=8):
        import torch
        import torch.nn as nn

        class _ChunkCVAEFeatureExtractor(base_extractor_cls):
            def __init__(self, observation_space, features_dim=104):
                super().__init__(observation_space, features_dim=features_dim)
                self._z_dim = z_dim
                self._obs_feature_dim = features_dim - z_dim

                self.state_mlp = nn.Sequential(
                    nn.Linear(10, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                )

                self.lidar_mlp = nn.Sequential(
                    nn.Linear(24, 128),
                    nn.SiLU(),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                )

            def encode_obs(self, observations):
                """Encode observations into obs_features (without z padding).

                Returns:
                    obs_features tensor of shape (batch, 96)
                """
                state = symlog(observations[:, :10])
                lidar = symlog(observations[:, 10:34])
                state_features = self.state_mlp(state)
                lidar_features = self.lidar_mlp(lidar)
                return torch.cat([state_features, lidar_features], dim=-1)

            def forward(self, observations):
                obs_features = self.encode_obs(observations)
                z_pad = torch.zeros(
                    obs_features.shape[0], self._z_dim,
                    device=obs_features.device, dtype=obs_features.dtype,
                )
                return torch.cat([obs_features, z_pad], dim=-1)

        return _ChunkCVAEFeatureExtractor

    @classmethod
    def get_class(cls, base_extractor_cls, z_dim=8):
        # Always recreate to capture z_dim closure
        cls._cls = cls.create(base_extractor_cls, z_dim=z_dim)
        return cls._cls


def pretrain_chunk_cvae(model, demo_obs, demo_actions, episode_lengths,
                        chunk_size, z_dim=8, epochs=100, batch_size=256,
                        lr=1e-3, beta=0.1, gamma=0.99):
    """Pretrain actor via Chunk CVAE: encoder maps (obs, action_chunk) → z,
    decoder (= actor's latent_pi + mu) maps (obs_features || z) → action_chunk.

    After pretraining the encoder is discarded. The z-slot is zeroed during RL.

    Args:
        model: SB3 SAC/TQC model (actor must use ChunkCVAEFeatureExtractor)
        demo_obs: numpy (N, 34) step-level observations
        demo_actions: numpy (N, 2) step-level actions
        episode_lengths: numpy array of per-episode step counts
        chunk_size: action chunk size (k)
        z_dim: CVAE latent dimension
        epochs: pretraining epochs
        batch_size: mini-batch size
        lr: learning rate
        beta: KL weight
        gamma: discount factor (unused here, kept for API consistency)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from demo_utils import extract_action_chunks

    print("\n" + "=" * 60)
    print("Chunk CVAE Pretraining")
    print("=" * 60)

    # Extract action chunks from demo data
    chunk_obs, chunk_actions_flat = extract_action_chunks(
        demo_obs, demo_actions, episode_lengths, chunk_size)
    action_chunk_dim = chunk_size * demo_actions.shape[1]

    print(f"  Chunk size: {chunk_size}, z_dim: {z_dim}")
    print(f"  {len(chunk_obs)} chunks from {len(episode_lengths)} episodes")

    device = model.device
    dataset = TensorDataset(
        torch.tensor(chunk_obs, dtype=torch.float32),
        torch.tensor(chunk_actions_flat, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get obs feature dim from the feature extractor
    obs_feature_dim = model.actor.features_extractor._obs_feature_dim

    # Temporary CVAE encoder (discarded after pretraining)
    cvae_encoder = nn.Sequential(
        nn.Linear(obs_feature_dim + action_chunk_dim, 128),
        nn.SiLU(),
        nn.Linear(128, 64),
        nn.SiLU(),
    ).to(device)
    cvae_mu = nn.Linear(64, z_dim).to(device)
    cvae_logvar = nn.Linear(64, z_dim).to(device)

    # Parameters to optimize: feature_extractor + latent_pi + mu + cvae_encoder
    params = (
        list(model.actor.features_extractor.parameters())
        + list(model.actor.latent_pi.parameters())
        + list(model.actor.mu.parameters())
        + list(cvae_encoder.parameters())
        + list(cvae_mu.parameters())
        + list(cvae_logvar.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    print(f"  Training CVAE for {epochs} epochs...")

    for epoch in range(epochs):
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            # Encode observations → obs_features (96D)
            obs_features = model.actor.features_extractor.encode_obs(obs_batch)

            # CVAE encoder: (obs_features, action_chunk) → z
            enc_input = torch.cat([obs_features, act_batch], dim=-1)
            h = cvae_encoder(enc_input)
            mu_z = cvae_mu(h)
            logvar_z = cvae_logvar(h)

            # Reparameterize
            std_z = torch.exp(0.5 * logvar_z)
            eps = torch.randn_like(std_z)
            z = mu_z + eps * std_z

            # Replace z-slot in features: concat(obs_features, z)
            features = torch.cat([obs_features, z], dim=-1)

            # Decode through actor's latent_pi → mu → tanh
            latent = model.actor.latent_pi(features)
            mean_actions = model.actor.mu(latent)
            pred_actions = torch.tanh(mean_actions)

            # Loss: L1 reconstruction + β·KL
            recon_loss = torch.nn.functional.l1_loss(pred_actions, act_batch)
            kl_loss = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}, "
                  f"L1: {total_recon / n_batches:.6f}, "
                  f"KL: {total_kl / n_batches:.6f}")

    # Copy pretrained features_extractor weights → critic and critic_target
    fe_state = model.actor.features_extractor.state_dict()
    model.critic.features_extractor.load_state_dict(fe_state)
    model.critic_target.features_extractor.load_state_dict(fe_state)
    print("  Feature extractor weights copied to critic/critic_target")

    # Tighten log_std to preserve CVAE-learned behavior
    model.actor.log_std.weight.data.zero_()
    model.actor.log_std.bias.data.fill_(-2.0)
    print("  Exploration noise tightened (log_std bias = -2.0, std ~ 0.135)")

    print("Chunk CVAE pretraining complete!")
    print("=" * 60 + "\n")


def make_demo_replay_buffer(buffer_cls, buffer_size, observation_space, action_space,
                            device, demo_obs, demo_actions, demo_rewards,
                            demo_next_obs, demo_dones, demo_ratio=0.5):
    """Create a replay buffer that mixes demos and online data at a given ratio.

    Returns a subclass instance of the given buffer_cls that overrides sample()
    to mix demo and online transitions.
    """
    import torch as th
    from stable_baselines3.common.buffers import ReplayBufferSamples

    class DemoReplayBuffer(buffer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Store demo data as tensors on device
            self.demo_obs = th.tensor(demo_obs, device=device)
            self.demo_actions = th.tensor(demo_actions, device=device)
            self.demo_rewards = th.tensor(demo_rewards, device=device).unsqueeze(1)
            self.demo_next_obs = th.tensor(demo_next_obs, device=device)
            self.demo_dones = th.tensor(demo_dones, device=device).unsqueeze(1)
            self.n_demos = len(demo_obs)

        def sample(self, batch_size, env=None):
            # When online buffer is empty, use 100% demo
            if self.size() == 0:
                demo_batch_size = batch_size
                online_batch_size = 0
            else:
                demo_batch_size = int(batch_size * demo_ratio)
                online_batch_size = batch_size - demo_batch_size

            # Sample demo indices
            demo_idx = np.random.randint(0, self.n_demos, size=demo_batch_size)
            demo_samples = ReplayBufferSamples(
                observations=self.demo_obs[demo_idx],
                actions=self.demo_actions[demo_idx],
                next_observations=self.demo_next_obs[demo_idx],
                dones=self.demo_dones[demo_idx],
                rewards=self.demo_rewards[demo_idx],
            )

            if online_batch_size == 0:
                return demo_samples

            # Sample from online buffer
            online_samples = super().sample(online_batch_size, env=env)

            # Concatenate
            return ReplayBufferSamples(
                observations=th.cat([demo_samples.observations, online_samples.observations]),
                actions=th.cat([demo_samples.actions, online_samples.actions]),
                next_observations=th.cat([demo_samples.next_observations, online_samples.next_observations]),
                dones=th.cat([demo_samples.dones, online_samples.dones]),
                rewards=th.cat([demo_samples.rewards, online_samples.rewards]),
            )

    # Instantiate
    buf = DemoReplayBuffer(
        buffer_size,
        observation_space,
        action_space,
        device=device,
    )
    return buf


def inject_layernorm_into_critics(model):
    """Post-hoc inject LayerNorm + OFN into critic networks.

    Injects LayerNorm after each hidden Linear layer and Output Feature
    Normalization (OFN) before the final output Linear layer.

    Handles both TQC (quantile_critics) and SAC (critic.qf*) structures.
    After injection, re-syncs critic_target and recreates the critic optimizer.
    """
    import torch
    import torch.nn as nn

    class OutputFeatureNorm(nn.Module):
        """L2-normalize features: x / ||x||_2 (RLC 2024 OFN)."""
        def forward(self, x):
            return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

    def _inject_norms(sequential):
        """Insert LayerNorm after hidden Linears and OFN before output Linear."""
        new_modules = []
        modules = list(sequential)
        for i, module in enumerate(modules):
            if isinstance(module, nn.Linear):
                remaining = modules[i + 1:]
                has_more_linear = any(isinstance(m, nn.Linear) for m in remaining)
                if has_more_linear:
                    # Hidden Linear: append layer, then LayerNorm
                    new_modules.append(module)
                    new_modules.append(nn.LayerNorm(module.out_features))
                else:
                    # Output Linear: insert OFN before it
                    new_modules.append(OutputFeatureNorm())
                    new_modules.append(module)
            else:
                new_modules.append(module)
        return nn.Sequential(*new_modules)

    is_tqc = hasattr(model.critic, 'quantile_critics')

    if is_tqc:
        for i, critic_net in enumerate(model.critic.quantile_critics):
            model.critic.quantile_critics[i] = _inject_norms(critic_net)
        for i, critic_net in enumerate(model.critic_target.quantile_critics):
            model.critic_target.quantile_critics[i] = _inject_norms(critic_net)
    else:
        qf_attrs = sorted(a for a in dir(model.critic) if a.startswith('qf') and a[2:].isdigit())
        for attr in qf_attrs:
            setattr(model.critic, attr, _inject_norms(getattr(model.critic, attr)))
            setattr(model.critic_target, attr, _inject_norms(getattr(model.critic_target, attr)))

    # Move new parameters to device
    model.critic = model.critic.to(model.device)
    model.critic_target = model.critic_target.to(model.device)

    # Sync target from critic weights
    model.critic_target.load_state_dict(model.critic.state_dict())

    # Recreate critic optimizer to include LayerNorm + OFN parameters
    model.critic.optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=model.lr_schedule(1)
    )

    print("LayerNorm + OFN injected into critic networks")


def main():
    parser = argparse.ArgumentParser(
        description='Train SAC/TQC agent with Chunk CVAE + Q-chunking for Jetbot navigation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demos demos/recording.npz --headless
      Train headless with demo replay

  %(prog)s --demos demos/recording.npz --headless --timesteps 500000
      Train for 500k timesteps

  %(prog)s --demos demos/recording.npz --headless --chunk-size 5
      Use chunk size 5 instead of default 10

  %(prog)s --demos demos/recording.npz --headless --cpu --timesteps 1000
      Quick smoke test on CPU
        """
    )

    # Training arguments
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 500000)')
    parser.add_argument('--demos', type=str, required=True,
                        help='Path to demo .npz file (required)')
    parser.add_argument('--utd-ratio', type=int, default=20,
                        help='Update-to-data ratio / gradient steps per env step (default: 20)')
    parser.add_argument('--buffer-size', type=int, default=300000,
                        help='Replay buffer size (default: 300000)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient (default: 0.005)')
    parser.add_argument('--ent-coef', type=str, default='auto_0.006',
                        help='Entropy coefficient, "auto" or "auto_<init>" for learned (default: auto_0.006)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--learning-starts', type=int, default=0,
                        help='Steps before training starts (default: 0, demos available immediately)')
    parser.add_argument('--demo-ratio', type=float, default=0.5,
                        help='Fraction of each batch sampled from demos, 0.0-1.0 (default: 0.5)')

    # Chunk CVAE arguments
    parser.add_argument('--chunk-size', type=int, default=10,
                        help='Action chunk size for Q-chunking (default: 10)')
    parser.add_argument('--cvae-z-dim', type=int, default=8,
                        help='CVAE latent dimension (default: 8)')
    parser.add_argument('--cvae-epochs', type=int, default=100,
                        help='CVAE pretraining epochs (default: 100)')
    parser.add_argument('--cvae-beta', type=float, default=0.1,
                        help='CVAE KL weight (default: 0.1)')
    parser.add_argument('--cvae-lr', type=float, default=1e-3,
                        help='CVAE pretraining learning rate (default: 1e-3)')

    # Environment arguments
    parser.add_argument('--reward-mode', choices=['dense', 'sparse'], default='dense',
                        help='Reward mode (default: dense)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (faster training)')
    parser.add_argument('--num-obstacles', type=int, default=5,
                        help='Number of obstacles to spawn (default: 5)')
    parser.add_argument('--arena-size', type=float, default=4.0,
                        help='Side length of square arena in meters (default: 4.0)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force training on CPU instead of GPU')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--min-goal', type=float, default=0.5,
                        help='Minimum distance from robot start to goal in meters (default: 0.5)')
    parser.add_argument('--inflation-radius', type=float, default=0.08,
                        help='Obstacle inflation radius for A* planner in meters (default: 0.08)')

    # Checkpoint arguments
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                        help='Save checkpoint every N steps (default: 50000)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .zip to resume training from')

    # Logging arguments
    parser.add_argument('--more-debug', action='store_true',
                        help='Print per-episode stats')

    # Output arguments
    parser.add_argument('--output', type=str, default='models/tqc_jetbot.zip',
                        help='Output model path (default: models/tqc_jetbot.zip)')
    parser.add_argument('--tensorboard-log', type=str, default='./runs/',
                        help='TensorBoard log directory (default: ./runs/)')

    args = parser.parse_args()

    # Validate demo_ratio
    if not 0.0 <= args.demo_ratio <= 1.0:
        parser.error(f"--demo-ratio must be between 0.0 and 1.0, got {args.demo_ratio}")

    # Parse ent_coef
    ent_coef = args.ent_coef
    if not ent_coef.startswith('auto'):
        ent_coef = float(ent_coef)

    print("=" * 60)
    print("SAC/TQC + Chunk CVAE + Q-Chunking for Jetbot Navigation")
    print("=" * 60)
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Demos: {args.demos}")
    print(f"  UTD ratio: {args.utd_ratio}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma} (effective: {args.gamma ** args.chunk_size:.6f})")
    print(f"  Tau: {args.tau}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Demo ratio: {args.demo_ratio}")
    print(f"  Learning starts: {args.learning_starts}")
    print(f"  CVAE: z_dim={args.cvae_z_dim}, epochs={args.cvae_epochs}, "
          f"beta={args.cvae_beta}, lr={args.cvae_lr}")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Headless: {args.headless}")
    print(f"  Seed: {args.seed}")
    print(f"  Max steps/episode: {args.max_steps}")
    print(f"  Output: {args.output}")
    print(f"  TensorBoard: {args.tensorboard_log}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    print("=" * 60 + "\n")

    # Validate demo data first (fail fast)
    validate_demo_data(args.demos)

    # Import here to allow --help without Isaac Sim
    import torch
    from jetbot_rl_env import JetbotNavigationEnv, ChunkedEnvWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from demo_utils import make_chunk_transitions

    # Try TQC first, fall back to SAC
    try:
        from sb3_contrib import TQC
        algo_cls = TQC
        algo_name = "TQC"
    except ImportError:
        from stable_baselines3 import SAC
        algo_cls = SAC
        algo_name = "SAC"
        print("sb3-contrib not found, falling back to SAC")

    print(f"Using algorithm: {algo_name}")

    # Create environment with ChunkedEnvWrapper
    import time as _time
    print("\nCreating environment...")
    _t0 = _time.time()
    half = args.arena_size / 2.0
    workspace_bounds = {'x': [-half, half], 'y': [-half, half]}
    raw_env = JetbotNavigationEnv(
        reward_mode=args.reward_mode,
        headless=args.headless,
        num_obstacles=args.num_obstacles,
        workspace_bounds=workspace_bounds,
        max_episode_steps=args.max_steps,
        min_goal_dist=args.min_goal,
        inflation_radius=args.inflation_radius,
    )
    raw_env = ChunkedEnvWrapper(raw_env, chunk_size=args.chunk_size, gamma=args.gamma)
    print(f"  Environment created in {_time.time() - _t0:.1f}s")
    print(f"  Observation space: {raw_env.observation_space.shape}")
    print(f"  Action space: {raw_env.action_space.shape} (chunk_size={args.chunk_size})")

    env = DummyVecEnv([lambda: raw_env])
    print("  No VecNormalize (using LayerNorm in critics instead)")
    print()

    # Load step-level demo transitions (for CVAE pretraining)
    print("Loading demo transitions...")
    demo_obs_step, demo_actions_step, demo_rewards_step, _, demo_dones_step = \
        load_demo_transitions(args.demos)
    demo_data = np.load(args.demos, allow_pickle=True)
    episode_lengths = demo_data['episode_lengths']

    # Build chunk-level transitions (for replay buffer)
    print("Building chunk-level transitions...")
    chunk_obs, chunk_acts, chunk_rews, chunk_next, chunk_dones = make_chunk_transitions(
        demo_obs_step, demo_actions_step, demo_rewards_step, demo_dones_step,
        episode_lengths, args.chunk_size, args.gamma)
    print(f"  {len(chunk_obs)} chunk transitions from {len(episode_lengths)} episodes")
    print()

    # Create replay buffer with chunk-level demo data
    device_str = "cpu" if args.cpu else "auto"
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    replay_buffer = make_demo_replay_buffer(
        ReplayBuffer,
        args.buffer_size,
        raw_env.observation_space,
        raw_env.action_space,
        device=device,
        demo_obs=chunk_obs,
        demo_actions=chunk_acts,
        demo_rewards=chunk_rews,
        demo_next_obs=chunk_next,
        demo_dones=chunk_dones,
        demo_ratio=args.demo_ratio,
    )
    print(f"Demo replay buffer created: {len(chunk_obs)} chunk-level demo transitions")
    print()

    # Effective gamma for chunk-level Bellman updates
    effective_gamma = args.gamma ** args.chunk_size

    # Feature extractor dimensions
    obs_feature_dim = 96
    features_dim = obs_feature_dim + args.cvae_z_dim

    # Create or resume model
    _t0 = _time.time()
    if args.resume:
        print(f"Resuming {algo_name} model from {args.resume}...")
        model = algo_cls.load(
            args.resume,
            env=env,
            device=device_str,
            tensorboard_log=args.tensorboard_log,
        )
        # Verify chunk size matches
        loaded_chunk = model.action_space.shape[0] // 2
        if loaded_chunk != args.chunk_size:
            print(f"  Warning: loaded model chunk_size={loaded_chunk} != --chunk-size={args.chunk_size}")
            print(f"  Using loaded chunk_size={loaded_chunk}")
        # Override mutable hyperparams the user may have changed
        model.learning_rate = args.lr
        model.batch_size = args.batch_size
        model.gamma = effective_gamma
        model.tau = args.tau
        model.gradient_steps = args.utd_ratio
        model.ent_coef = ent_coef
        # Replace replay buffer with fresh demo buffer (online data is lost)
        model.replay_buffer = replay_buffer
        # LayerNorm + CVAE weights are already baked into the loaded checkpoint
        print(f"  Model resumed in {_time.time() - _t0:.1f}s")
    else:
        print(f"Creating {algo_name} model...")
        fe_cls = ChunkCVAEFeatureExtractor.get_class(
            BaseFeaturesExtractor, z_dim=args.cvae_z_dim)
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            activation_fn=torch.nn.ReLU,
            features_extractor_class=fe_cls,
            features_extractor_kwargs=dict(features_dim=features_dim),
        )
        if algo_name == "TQC":
            policy_kwargs['n_critics'] = 5

        model = algo_cls(
            "MlpPolicy",
            env,
            verbose=1,
            device=device_str,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=effective_gamma,
            tau=args.tau,
            ent_coef=ent_coef,
            target_entropy=-2.0,
            gradient_steps=args.utd_ratio,
            learning_starts=args.learning_starts,
            train_freq=1,
            policy_kwargs=policy_kwargs,
        )
        # Replace the default replay buffer with our demo buffer
        model.replay_buffer = replay_buffer
        # Inject LayerNorm into critics
        inject_layernorm_into_critics(model)
        print(f"  Model created in {_time.time() - _t0:.1f}s")
    print()

    # CVAE pretraining (replaces BC warmstart)
    if not args.resume:
        pretrain_chunk_cvae(
            model, demo_obs_step, demo_actions_step, episode_lengths,
            chunk_size=args.chunk_size, z_dim=args.cvae_z_dim,
            epochs=args.cvae_epochs, batch_size=args.batch_size,
            lr=args.cvae_lr, beta=args.cvae_beta, gamma=args.gamma,
        )

    # Create callbacks
    checkpoint_dir = Path(args.output).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prefix = "tqc_jetbot" if algo_name == "TQC" else "sac_jetbot"
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=prefix,
        verbose=1,
    )
    callbacks = [checkpoint_callback]
    if args.more_debug:
        callbacks.append(VerboseEpisodeCallback.create(BaseCallback))
    callback = CallbackList(callbacks)

    # Train
    print("\n" + "=" * 60)
    print(f"Starting {algo_name} + Chunk CVAE + Q-Chunking Training")
    print("=" * 60)
    print(f"View TensorBoard: tensorboard --logdir {args.tensorboard_log}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=not args.resume,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")

    # Save final model (no VecNormalize stats to save)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Model saved to: {output_path}")
    print(f"  Checkpoints in: {checkpoint_dir}")
    print(f"  TensorBoard logs: {args.tensorboard_log}")
    print("\nTo evaluate the trained policy:")
    print(f"  ./run.sh eval_policy.py {output_path}")
    print("=" * 60)

    env.close()


if __name__ == '__main__':
    main()
