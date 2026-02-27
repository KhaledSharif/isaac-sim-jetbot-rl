"""Diagnostic tool for inspecting SB3 model checkpoint (.zip) files.

Usage:
    python src/inspect_checkpoint.py models/crossq_jetbot.zip
    ./run.sh inspect_checkpoint.py models/crossq_jetbot.zip

No Isaac Sim dependency — pure SB3/PyTorch.
"""

import argparse
import math
import sys


def _fmt(val, width=9):
    """Format a float for table display."""
    if val is None:
        return "N/A".rjust(width)
    try:
        return f"{val:+{width}.4f}"
    except (TypeError, ValueError):
        return str(val).rjust(width)


def _tensor_stats(t):
    """Return (min, max, mean, std, l2_norm, n_nan, n_inf) for a torch tensor."""
    import torch
    t = t.float()
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return None, None, None, None, None, n_nan, n_inf
    mn = finite.min().item()
    mx = finite.max().item()
    mean = finite.mean().item()
    std = finite.std().item() if finite.numel() > 1 else 0.0
    l2 = t.norm(p=2).item()
    return mn, mx, mean, std, l2, n_nan, n_inf


def _group_key(name):
    """Return a sort key for grouping tensor names."""
    n = name.lower()
    if "actor" in n:
        return (0, name)
    if "critic_target" in n or "target" in n:
        return (2, name)
    if "cost_critic" in n:
        return (3, name)
    if "critic" in n:
        return (1, name)
    if "features_extractor" in n or "feature" in n:
        return (4, name)
    if "optimizer" in n or "opt" in n:
        return (5, name)
    return (6, name)


def _print_section(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _describe_space(space):
    """Return a short string describing a gymnasium/gym space."""
    cls = type(space).__name__
    if hasattr(space, 'shape'):
        return f"{cls}  shape={space.shape}  dtype={getattr(space, 'dtype', '?')}"
    return cls


def inspect_checkpoint(path: str):
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
    except ImportError:
        print("ERROR: stable_baselines3 not found. Install with: pip install stable-baselines3")
        sys.exit(1)

    try:
        import torch
    except ImportError:
        print("ERROR: torch not found. Install PyTorch first.")
        sys.exit(1)

    print(f"\nInspecting checkpoint: {path}")

    try:
        data, params, pytorch_vars = load_from_zip_file(path, device="cpu")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        sys.exit(1)

    warnings = []

    # ------------------------------------------------------------------ #
    # Section 1: Hyperparameters
    # ------------------------------------------------------------------ #
    _print_section("1. Hyperparameters")

    # Policy & env
    policy_cls = data.get("policy_class", "?")
    n_envs = data.get("n_envs", "?")
    print(f"  policy_class     : {policy_cls}")
    print(f"  n_envs           : {n_envs}")

    # Observation / action spaces
    obs_space = data.get("observation_space", None)
    act_space = data.get("action_space", None)

    obs_dim = None
    obs_shape = None
    n_frames = None
    if obs_space is not None:
        obs_shape = getattr(obs_space, 'shape', None)
        print(f"  observation_space: {_describe_space(obs_space)}")
        if obs_shape:
            obs_total = obs_shape[0]
            obs_dim = obs_total
            # Determine base obs_dim and n_frames
            for base in (36, 34):
                if obs_total % base == 0:
                    n_frames = obs_total // base
                    obs_dim = base
                    break
            print(f"    -> obs_dim={obs_dim}  n_frames={n_frames if n_frames else 1}")
    else:
        print("  observation_space: (not found)")

    chunk_size = None
    if act_space is not None:
        act_shape = getattr(act_space, 'shape', None)
        print(f"  action_space     : {_describe_space(act_space)}")
        if act_shape:
            chunk_size = act_shape[0] // 2
            print(f"    -> chunk_size={chunk_size}  (action_dim={act_shape[0]})")
    else:
        print("  action_space     : (not found)")

    # Standard SB3 hyperparams
    for key in ("gamma", "tau", "learning_rate", "batch_size", "buffer_size",
                "gradient_steps", "policy_delay"):
        val = data.get(key, None)
        if val is not None:
            print(f"  {key:<20}: {val}")

    # policy_kwargs (pretty print)
    pk = data.get("policy_kwargs", None)
    if pk:
        print(f"  policy_kwargs    :")
        if isinstance(pk, dict):
            for k, v in pk.items():
                print(f"    {k}: {v}")
        else:
            print(f"    {pk}")

    # Anomaly checks on hyperparams
    if chunk_size is not None and chunk_size not in {5, 10, 25}:
        warnings.append(f"NOTE:    chunk_size={chunk_size} (unusual; common values: 5, 10, 25)")
    if obs_dim is not None and obs_dim not in (34, 36):
        warnings.append(f"NOTE:    obs_dim={obs_dim} (expected 34 or 36)")

    # ------------------------------------------------------------------ #
    # Section 2: PyTorch scalar variables
    # ------------------------------------------------------------------ #
    _print_section("2. PyTorch Scalar Variables")

    if not pytorch_vars:
        print("  (none)")
    else:
        for var_name, tensor in pytorch_vars.items():
            try:
                raw_val = tensor.item() if hasattr(tensor, 'item') else float(tensor)
            except Exception:
                raw_val = tensor

            if "log_ent_coef" in var_name:
                ent_coef = math.exp(raw_val)
                print(f"  {var_name:<30}: {raw_val:+.6f}  ->  ent_coef = {ent_coef:.6f}")
                if ent_coef < 0.005:
                    warnings.append(
                        f"WARNING: ent_coef={ent_coef:.4f} < 0.005 (entropy death spiral risk)"
                    )
                elif ent_coef > 5.0:
                    warnings.append(
                        f"WARNING: ent_coef={ent_coef:.4f} > 5.0 (very high, policy may be near-random)"
                    )
            elif "log_lagrange" in var_name or "lagrange" in var_name.lower():
                lam = math.exp(raw_val)
                print(f"  {var_name:<30}: {raw_val:+.6f}  ->  lambda = {lam:.6f}")
            else:
                print(f"  {var_name:<30}: {raw_val}")

    # ------------------------------------------------------------------ #
    # Section 3: Network weight statistics
    # ------------------------------------------------------------------ #
    _print_section("3. Network Weight Statistics")

    if not params:
        print("  (no params found)")
    else:
        # SB3 load_from_zip_file returns a nested dict:
        #   params['policy']           -> OrderedDict of {layer_name: tensor}
        #   params['actor.optimizer']  -> {'state': {...}, 'param_groups': [...]}
        #   params['critic.optimizer'] -> same structure
        #   etc.
        # We iterate over the nested state dicts for weight stats.

        # Collect all weight tensors into a flat dict
        flat_weights = {}
        optimizer_dicts = {}

        for top_key, value in params.items():
            if "optimizer" in top_key.lower():
                optimizer_dicts[top_key] = value
            elif hasattr(value, 'items'):
                # State dict (OrderedDict or plain dict of tensors)
                for sub_key, tensor in value.items():
                    if hasattr(tensor, 'shape'):
                        flat_weights[sub_key] = tensor
            elif hasattr(value, 'shape'):
                flat_weights[top_key] = value

        weight_keys = sorted(flat_weights.keys(), key=_group_key)

        # Table header
        col_name  = 50
        col_shape = 16
        col_dtype = 8
        col_val   = 10
        col_l2    = 11

        header = (
            f"{'Tensor Name':<{col_name}}"
            f"{'Shape':<{col_shape}}"
            f"{'dtype':<{col_dtype}}"
            f"{'min':>{col_val}}"
            f"{'max':>{col_val}}"
            f"{'mean':>{col_val}}"
            f"{'std':>{col_val}}"
            f"{'L2-norm':>{col_l2}}"
            f"  {'NaN':>4}"
            f"  {'Inf':>4}"
        )
        sep = "-" * len(header)
        print(header)
        print(sep)

        for name in weight_keys:
            tensor = flat_weights[name]
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype).replace("torch.", "")
            mn, mx, mean, std, l2, n_nan, n_inf = _tensor_stats(tensor)

            row = (
                f"{name[:col_name-1]:<{col_name}}"
                f"{str(shape):<{col_shape}}"
                f"{dtype:<{col_dtype}}"
                f"{_fmt(mn, col_val)}"
                f"{_fmt(mx, col_val)}"
                f"{_fmt(mean, col_val)}"
                f"{_fmt(std, col_val)}"
                f"{_fmt(l2, col_l2)}"
                f"  {n_nan:>4}"
                f"  {n_inf:>4}"
            )
            print(row)

            if n_nan > 0 or n_inf > 0:
                warnings.append(f"CRITICAL: NaN/Inf in {name}  (NaN={n_nan}, Inf={n_inf})")
            # Skip L2 warning for scalar step-counter tensors (e.g. BatchRenorm .steps)
            is_scalar_int = (tensor.ndim == 0 and not tensor.dtype.is_floating_point)
            if l2 is not None and l2 > 500 and not is_scalar_int:
                warnings.append(f"WARNING: large L2-norm in {name}: {l2:.1f}")

        # Optimizer summaries
        if optimizer_dicts:
            import numpy as _np
            print()
            print("  --- Optimizer state summaries ---")

            for opt_name, opt_state in sorted(optimizer_dicts.items()):
                state = opt_state.get("state", {})
                param_groups = opt_state.get("param_groups", [])
                n_tracked = len(state)

                exp_avg_norms = []
                exp_avg_sq_norms = []
                for param_idx, param_state in state.items():
                    if not isinstance(param_state, dict):
                        continue
                    for k, v in param_state.items():
                        if not hasattr(v, 'norm'):
                            continue
                        norm = v.float().norm(p=2).item()
                        if "exp_avg_sq" in k:
                            exp_avg_sq_norms.append(norm)
                        elif "exp_avg" in k:
                            exp_avg_norms.append(norm)

                print(f"  {opt_name}:")
                print(f"    tracked_params : {n_tracked}")
                if param_groups:
                    lr = param_groups[0].get("lr", "?")
                    print(f"    lr             : {lr}")
                if exp_avg_norms:
                    print(f"    exp_avg   norm : mean={_np.mean(exp_avg_norms):.3f}  max={_np.max(exp_avg_norms):.3f}")
                if exp_avg_sq_norms:
                    print(f"    exp_avg_sq norm: mean={_np.mean(exp_avg_sq_norms):.3f}  max={_np.max(exp_avg_sq_norms):.3f}")

    # ------------------------------------------------------------------ #
    # Section 4: Warnings
    # ------------------------------------------------------------------ #
    _print_section("4. Warnings / Anomaly Flags")

    if not warnings:
        print("  [OK] No anomalies detected")
    else:
        for w in warnings:
            print(f"  {w}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an SB3 model checkpoint (.zip) — no Isaac Sim required"
    )
    parser.add_argument("path", help="Path to .zip checkpoint file")
    args = parser.parse_args()
    inspect_checkpoint(args.path)


if __name__ == "__main__":
    main()
