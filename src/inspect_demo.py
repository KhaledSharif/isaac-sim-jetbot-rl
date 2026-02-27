"""Diagnostic tool for inspecting demo files (.npz or .hdf5).

Usage:
    python src/inspect_demo.py demos/recording.npz
    python src/inspect_demo.py demos/recording.hdf5
    ./run.sh inspect_demo.py demos/recording.npz

No Isaac Sim dependency — pure numpy/h5py.
"""

import argparse
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# OBS_VERSION=2 dimension names (34D base)
# ---------------------------------------------------------------------------

_OBS_NAMES_34 = [
    "pos_x_norm",    # 0
    "pos_y_norm",    # 1
    "sin_heading",   # 2
    "cos_heading",   # 3
    "linear_vel",    # 4
    "angular_vel",   # 5
    "goal_bf_x",     # 6
    "goal_bf_y",     # 7
    "dist_to_goal",  # 8
    "goal_reached",  # 9
] + [f"lidar_{i:02d}" for i in range(24)]  # 10-33

_OBS_NAMES_36 = (
    _OBS_NAMES_34[:10]
    + ["prev_lin_vel", "prev_ang_vel"]
    + [f"lidar_{i:02d}" for i in range(24)]
)

# Expected ranges for flag checks (min, max)
_EXPECTED_RANGES = {
    "pos_x_norm":   (0.0, 1.0),
    "pos_y_norm":   (0.0, 1.0),
    "sin_heading":  (-1.0, 1.0),
    "cos_heading":  (-1.0, 1.0),
    "goal_reached": (0.0, 1.0),
}
for _i in range(24):
    _EXPECTED_RANGES[f"lidar_{_i:02d}"] = (0.0, 1.0)


def _obs_names_for_dim(obs_dim: int):
    """Return dimension name list for a given obs_dim (handles frame stacking)."""
    if obs_dim == 34:
        return _OBS_NAMES_34
    if obs_dim == 36:
        return _OBS_NAMES_36
    # Frame-stacked: detect base dim
    for base, names in [(36, _OBS_NAMES_36), (34, _OBS_NAMES_34)]:
        if obs_dim % base == 0:
            n_frames = obs_dim // base
            result = []
            for f in range(n_frames):
                result.extend([f"[f{f}]{n}" for n in names])
            return result
    # Fallback
    return [f"dim_{i:03d}" for i in range(obs_dim)]


def _pct(arr, q):
    """Return percentile of a 1D array, safely."""
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _print_section(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def inspect_demo(path: str):
    # Import here so the script has no Isaac Sim dependency at module level
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from demo_io import open_demo
    except ImportError:
        print("ERROR: demo_io.py not found. Run from the repo root or src/ directory.")
        sys.exit(1)

    print(f"\nInspecting demo: {path}")

    try:
        data = open_demo(path)
    except Exception as e:
        print(f"ERROR: Failed to open demo file: {e}")
        sys.exit(1)

    warnings = []

    # ------------------------------------------------------------------ #
    # Section 1: File Metadata
    # ------------------------------------------------------------------ #
    _print_section("1. File Metadata")

    ext = os.path.splitext(path)[1].lower()
    fmt = "NPZ" if ext == ".npz" else "HDF5"
    print(f"  File      : {path}")
    print(f"  Format    : {fmt}")

    metadata = {}
    if "metadata" in data:
        raw = data["metadata"]
        if hasattr(raw, "item"):
            metadata = raw.item()
        elif isinstance(raw, dict):
            metadata = raw
        else:
            metadata = {}

    if metadata:
        for k, v in sorted(metadata.items()):
            print(f"  {k:<20}: {v}")
    else:
        print("  (no metadata)")

    obs_version   = metadata.get("obs_version", "?")
    arena_size    = metadata.get("arena_size", None)
    obs_dim_meta  = metadata.get("obs_dim", None)
    action_dim_meta = metadata.get("action_dim", None)
    has_cost      = metadata.get("has_cost", False)

    # ------------------------------------------------------------------ #
    # Load arrays
    # ------------------------------------------------------------------ #
    try:
        observations = data["observations"].astype(np.float32)
        actions      = data["actions"].astype(np.float32)
        rewards      = data["rewards"].astype(np.float32)
        dones        = data["dones"] if "dones" in data else np.zeros(len(observations), dtype=bool)
        costs        = data["costs"].astype(np.float32) if "costs" in data else None
    except Exception as e:
        print(f"ERROR: Failed to load arrays: {e}")
        sys.exit(1)

    episode_lengths = data["episode_lengths"].astype(np.int64)
    episode_returns_arr = data["episode_returns"].astype(np.float32) if "episode_returns" in data else None
    episode_success_arr = data["episode_success"] if "episode_success" in data else None

    n_episodes  = len(episode_lengths)
    total_steps = int(np.sum(episode_lengths))
    obs_dim     = observations.shape[1] if observations.ndim == 2 else 0

    # ------------------------------------------------------------------ #
    # Section 2: Episode-level Summary
    # ------------------------------------------------------------------ #
    _print_section("2. Episode-level Summary")

    # Compute per-episode outcomes
    n_success   = 0
    n_collision = 0
    n_truncated = 0
    ep_returns  = []
    ep_lengths  = episode_lengths.tolist()

    if episode_success_arr is not None:
        n_success = int(np.sum(episode_success_arr))

    # Recompute returns from rewards if episode_returns not available
    if episode_returns_arr is not None:
        ep_returns = episode_returns_arr.tolist()
    else:
        offset = 0
        for length in ep_lengths:
            ep_returns.append(float(np.sum(rewards[offset:offset + length])))
            offset += length

    # Infer collision and truncation from dones + success
    # collision = terminal done but NOT success (reward ~ -25 at last step)
    # truncation = not done (or done=0) and not success
    offset = 0
    for i, length in enumerate(ep_lengths):
        last_done = bool(dones[offset + length - 1]) if len(dones) > offset + length - 1 else False
        success = bool(episode_success_arr[i]) if episode_success_arr is not None else False
        last_reward = float(rewards[offset + length - 1]) if length > 0 else 0.0

        if not success:
            # Heuristic: collision if terminal done and last reward near -25
            if last_done and last_reward < -20.0:
                n_collision += 1
            else:
                n_truncated += 1
        offset += length

    sr = n_success / n_episodes * 100 if n_episodes > 0 else 0.0
    cr = n_collision / n_episodes * 100 if n_episodes > 0 else 0.0
    tr = n_truncated / n_episodes * 100 if n_episodes > 0 else 0.0

    print(f"  Success rate  : {sr:5.1f}%  ({n_success} / {n_episodes})")
    print(f"  Collision rate: {cr:5.1f}%  ({n_collision} / {n_episodes})")
    print(f"  Truncation rate: {tr:4.1f}%  ({n_truncated} / {n_episodes})")

    print()
    print("  Episode lengths:")
    lens_arr = np.array(ep_lengths, dtype=np.float64)
    p = np.percentile(lens_arr, [25, 50, 75])
    print(
        f"    min={int(lens_arr.min())}  p25={int(p[0])}  p50={int(p[1])}  "
        f"p75={int(p[2])}  max={int(lens_arr.max())}  "
        f"mean={lens_arr.mean():.1f}  std={lens_arr.std():.1f}"
    )

    if len(short_eps := [l for l in ep_lengths if l < 10]) > 0:
        warnings.append(
            f"WARNING: {len(short_eps)} episode(s) shorter than 10 steps (possible recording bug)"
        )

    print()
    print("  Episode returns:")
    if ep_returns:
        ret_arr = np.array(ep_returns, dtype=np.float64)
        p = np.percentile(ret_arr, [25, 50, 75])
        print(
            f"    min={ret_arr.min():.1f}  p25={p[0]:.1f}  p50={p[1]:.1f}  "
            f"p75={p[2]:.1f}  max={ret_arr.max():.1f}  "
            f"mean={ret_arr.mean():.1f}  std={ret_arr.std():.1f}"
        )

        # Outlier check: >3 sigma above mean
        ret_mean = ret_arr.mean()
        ret_std = ret_arr.std()
        if ret_std > 0:
            for i, r in enumerate(ep_returns):
                if r > ret_mean + 3 * ret_std:
                    warnings.append(
                        f"WARNING: Episode {i} return={r:.1f} (>3 sigma above "
                        f"mean={ret_mean:.1f}, sigma={ret_std:.1f})"
                    )

    # ------------------------------------------------------------------ #
    # Section 3: Per-array Global Statistics
    # ------------------------------------------------------------------ #
    _print_section("3. Per-Array Global Statistics")

    def _print_array_stats(name, arr):
        if arr is None:
            return
        shape = arr.shape
        total_count = arr.size
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        finite = arr[np.isfinite(arr)]

        print(f"\n  {name}  shape={shape}  dtype={arr.dtype}")
        print(f"    count={total_count:,}  NaN={n_nan}  Inf={n_inf}")

        if len(finite) == 0:
            print(f"    (all non-finite)")
            warnings.append(f"CRITICAL: All values non-finite in {name}")
            return

        p = np.percentile(finite, [1, 25, 50, 75, 99])
        print(
            f"    min={finite.min():.4f}  p01={p[0]:.4f}  p25={p[1]:.4f}  "
            f"p50={p[2]:.4f}  p75={p[3]:.4f}  p99={p[4]:.4f}  max={finite.max():.4f}"
        )
        print(f"    mean={finite.mean():.4f}  std={finite.std():.4f}")

        if n_nan > 0:
            warnings.append(f"CRITICAL: NaN values in {name}: {n_nan} entries")
        if n_inf > 0:
            warnings.append(f"CRITICAL: Inf values in {name}: {n_inf} entries")

    _print_array_stats("observations", observations)
    _print_array_stats("actions", actions)
    _print_array_stats("rewards", rewards)
    dones_float = dones.astype(np.float32) if dones is not None else None
    _print_array_stats("dones", dones_float)
    if has_cost and costs is not None:
        _print_array_stats("costs", costs)
    elif costs is not None and np.any(costs != 0):
        _print_array_stats("costs", costs)

    # ------------------------------------------------------------------ #
    # Section 4: Per-observation-dimension Statistics
    # ------------------------------------------------------------------ #
    _print_section("4. Per-Observation-Dimension Statistics")

    if obs_dim == 0:
        print("  (no observation data)")
    else:
        dim_names = _obs_names_for_dim(obs_dim)

        # Column widths
        c_dim  = 4
        c_name = 22
        c_val  = 9
        c_flag = 40

        header = (
            f"{'Dim':>{c_dim}} "
            f"{'Name':<{c_name}}"
            f"{'mean':>{c_val}}"
            f"{'std':>{c_val}}"
            f"{'min':>{c_val}}"
            f"{'max':>{c_val}}"
            f"{'p01':>{c_val}}"
            f"{'p99':>{c_val}}"
            f"  Flags"
        )
        sep = "-" * (c_dim + 1 + c_name + c_val * 6 + 2 + 20)
        print(header)
        print(sep)

        for i in range(obs_dim):
            col = observations[:, i]
            finite = col[np.isfinite(col)]
            n_nan = int(np.sum(np.isnan(col)))
            n_inf = int(np.sum(np.isinf(col)))

            name = dim_names[i] if i < len(dim_names) else f"dim_{i:03d}"
            # Strip frame prefix for range lookup
            bare_name = name.split("]")[-1] if "]" in name else name

            if len(finite) == 0:
                mean_ = std_ = mn_ = mx_ = p01_ = p99_ = float("nan")
            else:
                mean_ = float(finite.mean())
                std_  = float(finite.std())
                mn_   = float(finite.min())
                mx_   = float(finite.max())
                p01_  = float(np.percentile(finite, 1))
                p99_  = float(np.percentile(finite, 99))

            flags = []
            if n_nan > 0:
                flags.append(f"NaN={n_nan}")
                warnings.append(f"CRITICAL: NaN values in observations (dim {i} {name}): {n_nan}")
            if n_inf > 0:
                flags.append(f"Inf={n_inf}")
                warnings.append(f"CRITICAL: Inf values in observations (dim {i} {name}): {n_inf}")
            if std_ < 0.001 and not math.isnan(std_):
                flags.append("low-var")
                warnings.append(
                    f"WARNING: dim {i} ({name}) has near-zero variance: std={std_:.5f}"
                )
            if bare_name in _EXPECTED_RANGES:
                lo, hi = _EXPECTED_RANGES[bare_name]
                tol = 0.01
                if mn_ < lo - tol or mx_ > hi + tol:
                    flags.append(f"range!")
                    warnings.append(
                        f"WARNING: dim {i} ({name}) outside [{lo},{hi}]: "
                        f"min={mn_:.4f}, max={mx_:.4f}"
                    )

            flag_str = "  ".join(flags) if flags else ""
            row = (
                f"{i:>{c_dim}} "
                f"{name[:c_name]:<{c_name}}"
                f"{mean_:>{c_val}.4f}"
                f"{std_:>{c_val}.4f}"
                f"{mn_:>{c_val}.4f}"
                f"{mx_:>{c_val}.4f}"
                f"{p01_:>{c_val}.4f}"
                f"{p99_:>{c_val}.4f}"
                f"  {flag_str}"
            )
            print(row)

    # ------------------------------------------------------------------ #
    # Section 5: Anomaly / Sanity Checks
    # ------------------------------------------------------------------ #
    _print_section("5. Anomaly / Sanity Checks")

    if not warnings:
        print("  [OK] No anomalies detected")
    else:
        for w in warnings:
            print(f"  {w}")

    print()

    data.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a demo file (.npz or .hdf5) -- no Isaac Sim required"
    )
    parser.add_argument("path", help="Path to demo file (.npz or .hdf5)")
    args = parser.parse_args()
    inspect_demo(args.path)


if __name__ == "__main__":
    main()
